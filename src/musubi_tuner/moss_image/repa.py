"""Optional image REPA for MOSS, independently implemented for Musubi.

References: Yu et al., https://arxiv.org/abs/2410.06940 and SimpleTuner's
IMAGE_REPA.md / MOSS recipe. No SimpleTuner runtime or source is imported.
Clean cached latents supply the teacher image, as in SimpleTuner's image path.
Only the projection head is trained; DINO and the VAE are frozen.
"""
from collections import OrderedDict
import hashlib
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.moss_image.repa_config import validate_config

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


def load_encoder(path):
    # Use an existing dependency, without torch.hub code execution or timm.
    from transformers import AutoConfig, Dinov2Model
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    if config.model_type != "dinov2":
        raise ValueError("REPA needs a standard DINOv2 model folder/ID. Other architectures and "
                         "DINO register-token variants are not supported.")
    return Dinov2Model.from_pretrained(path, config=config).eval().requires_grad_(False)


class MossRepa:
    def __init__(self, args, hidden_size, training_device, encoder=None):
        validate_config(args)
        self.args = args
        self.training_device = torch.device(training_device)
        self.helper_device = self.training_device if args.moss_repa_device == "training" else torch.device("cpu")
        self.encoder = (encoder if encoder is not None else load_encoder(args.moss_repa_encoder)).eval().requires_grad_(False)
        self.projector = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.encoder.config.hidden_size))
        self.projector.to(self.training_device, dtype=torch.float32)
        self.cache = OrderedDict()
        self.config = {key: getattr(args, key) for key in
                       ("moss_repa_encoder", "moss_repa_block", "moss_repa_image_size")}
        self.config["hidden_size"] = hidden_size
        logger.info("MOSS REPA enabled: encoder=%s, block=%s, strength=%s, teacher_size=%s, helper_device=%s, "
                    "RAM cache=%s images. Frozen helpers return to CPU before the training forward.",
                    args.moss_repa_encoder, args.moss_repa_block, args.moss_repa_weight,
                    args.moss_repa_image_size, self.helper_device, args.moss_repa_cache_size)

    @staticmethod
    def cache_key(latent):
        cpu = latent.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
        return (str(cpu.dtype), tuple(cpu.shape), digest)

    @torch.no_grad()
    def targets(self, vae, latents):
        """Compute missing clean-image targets before DiT activations exist.

        The bounded CPU cache is per process/run and keyed by exact latent data;
        it cannot silently reuse a target after images, buckets or VAE change.
        Helpers return to CPU before the training forward, including on errors.
        """
        results = []
        try:
            for sample in latents:
                key = self.cache_key(sample)
                if key in self.cache:
                    result = self.cache.pop(key)
                    self.cache[key] = result
                else:
                    vae.to(self.helper_device).eval().requires_grad_(False)
                    with torch.autocast(device_type=self.helper_device.type, enabled=False):
                        pixels = vae.decode(sample.unsqueeze(0).squeeze(2).to(self.helper_device, dtype=torch.float32))
                        pixels = pixels.float().clamp(-1, 1).add(1).div(2)
                        pixels = F.interpolate(pixels, size=(self.args.moss_repa_image_size,) * 2,
                                               mode="bilinear", align_corners=False)
                    # The VAE and encoder never need to occupy GPU memory together.
                    vae.to("cpu")
                    self.encoder.to(device=self.helper_device, dtype=torch.float32)
                    with torch.autocast(device_type=self.helper_device.type, enabled=False):
                        mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                        std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                        features = self.encoder(pixel_values=(pixels - mean) / std, interpolate_pos_encoding=True,
                                                return_dict=True)
                        # Standard HF DINOv2: CLS followed by the square patch grid.
                        result = features.last_hidden_state[:, 1:].float().cpu().contiguous()
                    del pixels, features
                    if self.args.moss_repa_cache_size:
                        self.cache[key] = result
                        while len(self.cache) > self.args.moss_repa_cache_size:
                            self.cache.popitem(last=False)
                results.append(result)
        finally:
            vae.to("cpu")
            self.encoder.to("cpu")
            if self.helper_device.type == "cuda":
                clean_memory_on_device(self.helper_device)
        return torch.cat(results).to(self.training_device)

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
