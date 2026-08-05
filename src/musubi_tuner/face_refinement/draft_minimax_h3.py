"""Truncated differentiable MiniMax-H3 image sampling for DRaFT-K."""

from __future__ import annotations

import torch

from musubi_tuner.minimax_h3.image_sampling import decode_image_latent, sample_image_latent


def generate_differentiable_h3(
    model,
    vae,
    text_hidden_states: torch.Tensor,
    *,
    resolution: int,
    denoise_steps: int,
    draft_k: int,
    seed: int,
    device: torch.device,
    checkpoint_vae: bool = True,
) -> torch.Tensor:
    if not 1 <= int(draft_k) <= int(denoise_steps):
        raise ValueError("MiniMax-H3 DRaFT-K must be between 1 and the denoising step count")
    latent = sample_image_latent(
        model,
        text_hidden_states,
        width=resolution,
        height=resolution,
        steps=denoise_steps,
        seed=seed,
        shift=12.0,
        device=device,
        dtype=torch.bfloat16,
        enable_grad_from_step=denoise_steps - draft_k,
    )
    vae_dtype = next(vae.parameters()).dtype
    pixels = decode_image_latent(
        vae,
        latent.to(device=device, dtype=vae_dtype),
        checkpoint_decode=checkpoint_vae,
    )
    return pixels[:, :, 0]
