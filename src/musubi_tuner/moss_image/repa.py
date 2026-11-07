"""Optional image REPA for MOSS, independently implemented for Musubi.

References: Yu et al., https://arxiv.org/abs/2410.06940 and SimpleTuner's
IMAGE_REPA.md / MOSS recipe. No SimpleTuner runtime or source is imported.
Clean cached latents supply the teacher image, as in SimpleTuner's image path.
Only the projection head is trained; DINO and the VAE are frozen.
"""
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

from musubi_tuner.moss_image.repa_config import validate_config
from musubi_tuner.moss_image.repa_cache import MossRepaTargets, latent_key

logger = logging.getLogger(__name__)


def add_arguments(parser):
    group = parser.add_argument_group("MOSS experimental representation alignment (REPA)")
    group.add_argument("--moss_repa_enabled", action="store_true")
    group.add_argument("--moss_repa_weight", type=float, default=0.5)
    group.add_argument("--moss_repa_encoder", default="facebook/dinov2-small")
    group.add_argument("--moss_repa_block", type=int, default=8)
    group.add_argument("--moss_repa_image_size", type=int, default=518)
    group.add_argument("--moss_repa_device", choices=["training", "cpu"], default="training")
    group.add_argument("--moss_repa_cache_size", type=int, default=64)
    group.add_argument("--moss_repa_cache_dir", default="")


class MossRepa:
    def __init__(self, args, hidden_size, training_device, encoder=None, feature_cache=None):
        validate_config(args)
        self.args = args
        self.training_device = torch.device(training_device)
        self.feature_cache = feature_cache if feature_cache is not None else MossRepaTargets(args, training_device, encoder)
        self.projector = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.feature_cache.feature_dim))
        self.projector.to(self.training_device, dtype=torch.float32)
        self.config = {key: getattr(args, key) for key in
                       ("moss_repa_encoder", "moss_repa_block", "moss_repa_image_size")}
        self.config["hidden_size"] = hidden_size
        logger.info("MOSS REPA enabled: encoder=%s, block=%s, strength=%s, teacher_size=%s, "
                    "RAM cache=%s images. Prepared targets are read without loading the teacher during training.",
                    args.moss_repa_encoder, args.moss_repa_block, args.moss_repa_weight,
                    args.moss_repa_image_size, args.moss_repa_cache_size)

    cache_key = staticmethod(latent_key)

    @property
    def encoder(self):
        return self.feature_cache.encoder

    @property
    def cache(self):
        return self.feature_cache.cache

    @torch.no_grad()
    def targets(self, vae, latents):
        return self.feature_cache.targets(vae, latents)

    def loss(self, hidden, target, height, width):
        if hidden.shape[1] != height * width:
            raise ValueError("REPA requires the image tokens only, in their original spatial order.")
        side = math.isqrt(target.shape[1])
        if side * side != target.shape[1]:
            raise ValueError("REPA DINOv2 target must contain a square grid of patch features.")
        # Preserve the true H/W layout for rectangular buckets. Flattened 1D
        # interpolation would mix unrelated rows when aspect ratios differ.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projected = self.projector(hidden.float())
            grid = projected.transpose(1, 2).reshape(hidden.shape[0], -1, height, width)
            grid = F.interpolate(grid, size=(side, side), mode="bilinear", align_corners=False)
            projected = grid.flatten(2).transpose(1, 2)
            similarity = (F.normalize(projected, dim=-1) * F.normalize(target.detach().float(), dim=-1)).sum(-1).mean()
            # A constant shift of the paper's negative cosine objective makes
            # logging non-negative without changing gradients.
            loss = 1 - similarity
        return loss, {"loss/repa": loss.detach(), "repa/cosine_similarity": similarity.detach(),
                      "loss/repa_weighted": (loss.detach() * self.args.moss_repa_weight)}

    def state_dict(self):
        return {"config": self.config, "projector": self.projector.state_dict()}

    def load_state_dict(self, state):
        if state["config"] != self.config:
            raise ValueError("Saved REPA configuration differs. Restore its encoder, block and image size, "
                             "or continue from LoRA weights with a fresh optimizer.")
        self.projector.load_state_dict(state["projector"], strict=True)
