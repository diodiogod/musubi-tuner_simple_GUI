"""Differentiable depth-consistency anchor for image-model training."""

from __future__ import annotations

from collections import OrderedDict
import hashlib

import torch
import torch.nn.functional as F


def reconstruct_clean_latents(noisy_latents: torch.Tensor, velocity: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Recover x0 from rectified-flow x_t = x0 + t * velocity."""
    t = (timesteps.float() / 1000.0).view(-1, *([1] * (noisy_latents.ndim - 1)))
    return noisy_latents.float() - t * velocity.float()


class DepthAnchor:
    def __init__(self, model_id: str, device: torch.device, input_size: int = 518, grad_checkpoint: bool = True):
        from transformers import AutoModelForDepthEstimation

        self.device = device
        self.input_size = input_size
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval().requires_grad_(False)
        if grad_checkpoint and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
        self.target_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.target_cache_limit = 256

    def to(self, device: torch.device | str):
        """Move only live inference tensors; cached target depths intentionally stay on CPU."""
        self.device = torch.device(device)
        self.model.to(self.device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)
        return self

    def _predict(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = F.interpolate(pixels.float(), (self.input_size, self.input_size), mode="bicubic", align_corners=False)
        values = (pixels.clamp(0, 1) - self.mean) / self.std
        return self.model(pixel_values=values).predicted_depth.unsqueeze(1)

    @staticmethod
    def _normalize(depth: torch.Tensor) -> torch.Tensor:
        flat = depth.flatten(1)
        median = flat.median(dim=1).values.view(-1, 1, 1, 1)
        scale = (depth - median).abs().flatten(1).mean(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        return (depth - median) / scale

    @staticmethod
    def cache_key(latents: torch.Tensor) -> str:
        raw = latents.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy().tobytes()
        return hashlib.sha1(raw).hexdigest()

    def target_depth(self, target_pixels: torch.Tensor, cache_key: str | None = None) -> torch.Tensor:
        if cache_key is not None and cache_key in self.target_cache:
            cached = self.target_cache.pop(cache_key)
            self.target_cache[cache_key] = cached
            return cached.to(self.device)
        with torch.no_grad():
            target = self._normalize(self._predict(target_pixels))
        if cache_key is not None:
            self.target_cache[cache_key] = target.detach().cpu()
            while len(self.target_cache) > self.target_cache_limit:
                self.target_cache.popitem(last=False)
        return target

    def loss(
        self, predicted_pixels: torch.Tensor, target_pixels: torch.Tensor | None = None,
        grad_weight: float = 0.5, target_depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        predicted = self._normalize(self._predict(predicted_pixels))
        if target_depth is None:
            if target_pixels is None:
                raise ValueError("target_pixels or target_depth is required")
            target_depth = self.target_depth(target_pixels)
        target = target_depth.to(predicted.device)
        value_loss = F.l1_loss(predicted, target)
        pred_dx, pred_dy = predicted[..., 1:] - predicted[..., :-1], predicted[..., 1:, :] - predicted[..., :-1, :]
        target_dx, target_dy = target[..., 1:] - target[..., :-1], target[..., 1:, :] - target[..., :-1, :]
        gradient_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
        return value_loss + grad_weight * gradient_loss
