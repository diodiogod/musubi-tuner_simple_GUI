"""Optional post-update noise regularization for adapter weights.

This module is intentionally independent from model-family trainers.  The only
integration point is ``NetworkTrainer.on_post_optimizer_step`` so upstream
trainer updates do not need to carry the implementation itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WeightNoiseConfig:
    sigma: float
    mode: str = "relative"
    bound_norm: bool = False

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("weight-noise sigma must be non-negative")
        if self.mode not in {"relative", "absolute"}:
            raise ValueError("weight-noise mode must be 'relative' or 'absolute'")


class AdapterWeightNoise:
    """Inject Gaussian noise into trainable adapter parameters after updates."""

    def __init__(self, config: WeightNoiseConfig):
        self.config = config

    @torch.no_grad()
    def apply(self, parameters) -> dict[str, float]:
        noise_sq = 0.0
        weight_sq = 0.0
        tensors = 0

        for parameter in parameters:
            if not parameter.requires_grad or not parameter.is_floating_point():
                continue

            before_norm = torch.linalg.vector_norm(parameter.float())
            if self.config.mode == "relative":
                scale = self.config.sigma * parameter.float().square().mean().sqrt()
            else:
                scale = torch.as_tensor(self.config.sigma, device=parameter.device, dtype=torch.float32)
            if scale.item() == 0.0:
                weight_sq += before_norm.item() ** 2
                tensors += 1
                continue

            noise = torch.randn_like(parameter, dtype=torch.float32) * scale
            parameter.add_(noise.to(parameter.dtype))

            if self.config.bound_norm and before_norm.item() > 0.0:
                after_norm = torch.linalg.vector_norm(parameter.float())
                if after_norm.item() > 0.0:
                    parameter.mul_((before_norm / after_norm).to(parameter.dtype))

            noise_sq += torch.linalg.vector_norm(noise).item() ** 2
            weight_sq += before_norm.item() ** 2
            tensors += 1

        return {
            "regularization/weight_noise_norm": noise_sq**0.5,
            "regularization/adapter_weight_norm": weight_sq**0.5,
            "regularization/weight_noise_tensors": float(tensors),
        }
