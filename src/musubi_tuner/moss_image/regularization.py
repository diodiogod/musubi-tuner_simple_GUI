"""Optional depth consistency for MOSS's packed Flux.2 latent representation."""
import argparse
import torch

from musubi_tuner.perceptual.depth_anchor import DepthAnchor, reconstruct_clean_latents, resize_latents_for_depth_decode
from musubi_tuner.perceptual.depth_devices import resolve_depth_vae_device
from musubi_tuner.utils.device_utils import clean_memory_on_device


def add_arguments(parser):
    group = parser.add_argument_group("MOSS experimental depth consistency")
    group.add_argument("--depth_anchor_weight", type=float, default=0.0)
    group.add_argument("--depth_anchor_model", default="depth-anything/Depth-Anything-V2-Small-hf")
    group.add_argument("--depth_anchor_input_size", type=int, default=518)
    group.add_argument("--depth_anchor_gradient_weight", type=float, default=0.5)
    group.add_argument("--depth_anchor_grad_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--depth_anchor_vae_device", default="training")
    group.add_argument("--keep_depth_helpers_on_gpu", action="store_true")
    group.add_argument("--weight_noise_sigma", type=float, default=0.0)
    group.add_argument("--weight_noise_mode", choices=["relative", "absolute"], default="relative")
    group.add_argument("--weight_noise_bound_norm", action="store_true")


class MossDepthRegularizer:
    def __init__(self, args, training_device):
        self.args = args
        self.training_device = training_device
        self.vae_device = (torch.device("cpu") if str(args.depth_anchor_vae_device).strip().lower() == "cpu"
                           else resolve_depth_vae_device(args.depth_anchor_vae_device, training_device))
        self.anchor = DepthAnchor(args.depth_anchor_model, training_device,
                                  input_size=args.depth_anchor_input_size,
                                  grad_checkpoint=args.depth_anchor_grad_checkpoint)

    def decode(self, vae, latents):
        # Each packed cell is a 2x2 raw latent patch: effective pixel stride 16.
        latents = resize_latents_for_depth_decode(latents, self.args.depth_anchor_input_size, 16)
        with torch.autocast(device_type=self.vae_device.type, enabled=False):
            pixels = vae.decode(latents.squeeze(2).to(self.vae_device, dtype=torch.float32))
        return pixels.add(1).div(2).to(self.training_device)

    def loss(self, vae, latents, noisy, prediction, timesteps):
        vae.to(self.vae_device).requires_grad_(False)
        self.anchor.to(self.training_device)
        predicted = self.decode(vae, reconstruct_clean_latents(noisy, prediction, timesteps))
        targets = []
        for sample in latents:
            key = self.anchor.cache_key(sample)
            if key in self.anchor.target_cache:
                targets.append(self.anchor.target_depth(None, key))
            else:
                with torch.no_grad():
                    pixels = self.decode(vae, sample.unsqueeze(0))
                    targets.append(self.anchor.target_depth(pixels, key))
        return self.anchor.loss(predicted, grad_weight=self.args.depth_anchor_gradient_weight,
                                 target_depth=torch.cat(targets))

    def release_after_backward(self, vae):
        if not self.args.keep_depth_helpers_on_gpu:
            vae.to("cpu")
            self.anchor.to("cpu")
            clean_memory_on_device(self.vae_device)
        clean_memory_on_device(self.training_device)
