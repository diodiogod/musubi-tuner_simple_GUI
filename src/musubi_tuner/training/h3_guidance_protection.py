"""MiniMax-H3 guidance-distillation protection for LoRA training.

Technique reference: Ostris AI Toolkit's contrastive guidance loss for H3,
including sigma-dependent balancing (commits 183433ae and 1e1418b22c). This is an independent
Musubi adaptation for cached text states and the image-only H3 trainer.
"""

from __future__ import annotations

import torch

from musubi_tuner.utils import train_utils


def enabled(args) -> bool:
    return bool(getattr(args, "h3_guidance_distillation_protection", False))


def validate(scale: float) -> float:
    value = float(scale)
    if not torch.isfinite(torch.tensor(value)) or value < 1.0:
        raise ValueError("MiniMax-H3 guidance protection scale must be a finite number of at least 1.0")
    return value


def validate_sigma_min(value: float) -> float:
    value = float(value)
    if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
        raise ValueError("MiniMax-H3 Dynamic Sigma minimum must be a finite number from 0.0 to 1.0")
    return value


def base_sigma_from_model_timestep(args, timesteps: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Recover the pre-resolution-shift sigma used by upstream's guidance gate."""
    sigma = ((timesteps.detach().float() - 1.0) / 1000.0).clamp(0.0, 1.0)
    sampling = str(getattr(args, "timestep_sampling", "") or "")
    shift = None
    if sampling == "krea2_shift":
        h, w = latents.shape[-2:]
        mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=6400, y2=1.15)((h // 2) * (w // 2))
        shift = float(torch.exp(torch.tensor(mu)).item())
    elif sampling == "shift":
        shift = float(getattr(args, "discrete_flow_shift", 1.0))
    elif sampling == "h3_shifted_uniform":
        shift = 12.0
    if shift is not None:
        denominator = shift - (shift - 1.0) * sigma
        sigma = sigma / denominator.clamp_min(torch.finfo(sigma.dtype).eps)
    return sigma.clamp(0.0, 1.0)


def build_guided_target(
    unconditional_prediction: torch.Tensor,
    normal_flow_target: torch.Tensor,
    scale: float,
    sigma: torch.Tensor | None = None,
    schedule: str = "sigma",
) -> torch.Tensor:
    """Amplify the conditional direction away from the empty-prompt prediction."""

    scale = validate(scale)
    if schedule not in {"sigma", "constant"}:
        raise ValueError(f"Unsupported MiniMax-H3 guidance protection schedule: {schedule}")
    unconditional = unconditional_prediction.detach().float()
    target = normal_flow_target.detach().float()
    effective_scale: float | torch.Tensor = scale
    if schedule == "sigma":
        if sigma is None:
            raise ValueError("Sigma-dependent H3 guidance protection requires the current timestep sigma")
        sigma = sigma.detach().to(device=target.device, dtype=target.dtype)
        while sigma.ndim < target.ndim:
            sigma = sigma.unsqueeze(-1)
        effective_scale = 1.0 + (scale - 1.0) * sigma
    return (unconditional + effective_scale * (target - unconditional)).to(normal_flow_target.dtype).detach()
