# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted for the image-only downstream path from the MiniMax-H3 scheduler
# semantics in Hugging Face Diffusers PR #14355 and musubi-tuner PR #1018.
# AI Toolkit is used only as an independent image-decoding reference.

"""Reusable MiniMax-H3 still-image sampling and decoding helpers."""

from __future__ import annotations

from collections.abc import Callable

import torch


VIDEO_VAE_SPATIAL_RATIO = 16
IMAGE_ALIGNMENT = 32
DEFAULT_VIDEO_SHIFT = 12.0


def align_image_size(value: int, label: str = "size") -> int:
    value = int(value)
    if value < IMAGE_ALIGNMENT:
        raise ValueError(f"MiniMax-H3 {label} must be at least {IMAGE_ALIGNMENT}, got {value}")
    return value // IMAGE_ALIGNMENT * IMAGE_ALIGNMENT


def shift_sigma(base: torch.Tensor, shift: float = DEFAULT_VIDEO_SHIFT) -> torch.Tensor:
    shift = float(shift)
    if not 0.01 <= shift <= 100.0:
        raise ValueError(f"MiniMax-H3 video shift must be in [0.01,100.0], got {shift}")
    return shift * base / (1.0 + (shift - 1.0) * base)


def build_image_sigma_schedule(
    steps: int,
    *,
    shift: float = DEFAULT_VIDEO_SHIFT,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError(f"MiniMax-H3 sampling steps must be a positive integer, got {steps}")
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64, device=device)
    return shift_sigma(base, shift)


def initialize_image_noise(
    width: int,
    height: int,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    width = align_image_size(width, "width")
    height = align_image_size(height, "height")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shape = (1, 24, 1, height // VIDEO_VAE_SPATIAL_RATIO, width // VIDEO_VAE_SPATIAL_RATIO)
    return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu").to(device=device, dtype=dtype)


def sample_image_latent(
    transformer,
    text_hidden_states: torch.Tensor,
    *,
    width: int,
    height: int,
    steps: int = 28,
    seed: int = 42,
    shift: float = DEFAULT_VIDEO_SHIFT,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    step_callback: Callable[[int, int], None] | None = None,
    enable_grad_from_step: int | None = None,
) -> torch.Tensor:
    """Denoise one image latent with H3's native clean-minus-noise velocity.

    ``enable_grad_from_step`` is used by DRaFT refinement.  Earlier steps are
    detached and run without an autograd graph; the final K steps can remain
    differentiable while using the exact same sampler as previews.
    """
    device = torch.device(device)
    text = text_hidden_states.to(device=device, dtype=dtype)
    if text.ndim == 2:
        text = text.unsqueeze(0)
    if text.ndim != 3 or text.shape[0] != 1 or text.shape[-1] != 5120:
        raise ValueError(f"MiniMax-H3 image text states must be [1,L,5120], got {tuple(text.shape)}")

    latent = initialize_image_noise(width, height, seed=seed, device=device, dtype=dtype)
    sigmas = build_image_sigma_schedule(steps, shift=shift, device=device)
    grad_start = steps if enable_grad_from_step is None else max(0, min(int(enable_grad_from_step), steps))

    for index in range(steps):
        grad_enabled = index >= grad_start
        if grad_enabled and not latent.requires_grad:
            latent = latent.detach().requires_grad_(True)
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            sigma = sigmas[index].to(torch.float32)
            prediction = transformer.forward_image(latent, 1.0 - sigma, text)
            delta = (sigmas[index] - sigmas[index + 1]).to(latent)
            latent = latent + delta * prediction
        if not grad_enabled:
            latent = latent.detach()
        if step_callback is not None:
            step_callback(index + 1, steps)
    return latent


def decode_image_latent(
    vae,
    latent: torch.Tensor,
    *,
    checkpoint_decode: bool = False,
    single_frame: bool = False,
) -> torch.Tensor:
    """Decode one H3 image latent to ``[B,3,1,H,W]`` in ``[0,1]``.

    The released video decoder was trained on temporal chunks. Decoding a lone
    temporal token through its single-clip shortcut is visibly patchy, so the
    default path duplicates it to a two-token video and keeps the first frame.
    Image-specialized H3 VAEs trained for direct T=1 decoding set
    ``single_frame=True`` and receive the original temporal token unchanged.
    """
    if latent.ndim != 5 or tuple(latent.shape[:3]) != (1, 24, 1):
        raise ValueError(f"MiniMax-H3 image latent must be [1,24,1,H,W], got {tuple(latent.shape)}")
    decode_input = latent if single_frame else torch.cat((latent, latent), dim=2)

    def decode(value: torch.Tensor) -> torch.Tensor:
        return vae.decode(value)[:, :, :1]

    if checkpoint_decode and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint

        pixels = checkpoint(decode, decode_input, use_reentrant=False)
    else:
        pixels = decode(decode_input)
    return ((pixels.float().clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
