"""MiniMax-H3 guidance-distillation protection for LoRA training.

Technique reference: Ostris AI Toolkit's contrastive guidance loss for H3
(github.com/ostris/ai-toolkit, commit 183433ae). This is an independent
Musubi adaptation for cached text states and the image-only H3 trainer.
"""

from __future__ import annotations

import torch


def enabled(args) -> bool:
    return bool(getattr(args, "h3_guidance_distillation_protection", False))


def validate(scale: float) -> float:
    value = float(scale)
    if not torch.isfinite(torch.tensor(value)) or value < 1.0:
        raise ValueError("MiniMax-H3 guidance protection scale must be a finite number of at least 1.0")
    return value


def build_guided_target(
    unconditional_prediction: torch.Tensor,
    normal_flow_target: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Amplify the conditional direction away from the empty-prompt prediction."""

    scale = validate(scale)
    unconditional = unconditional_prediction.detach().float()
    target = normal_flow_target.detach().float()
    return (unconditional + scale * (target - unconditional)).to(normal_flow_target.dtype).detach()
