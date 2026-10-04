"""Truncated differentiable MiniMax-H3 image sampling for DRaFT-K."""

from __future__ import annotations

import torch

from musubi_tuner.minimax_h3.image_sampling import decode_image_latent, sample_image_latent
from musubi_tuner.minimax_h3.video_sampling import sample_video_latent


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
    training_frame_mode: str = "one_frame",
) -> torch.Tensor:
    if not 1 <= int(draft_k) <= int(denoise_steps):
        raise ValueError("MiniMax-H3 DRaFT-K must be between 1 and the denoising step count")
    if training_frame_mode not in {"one_frame", "five_frame"}:
        raise ValueError("MiniMax-H3 training frame mode must be one_frame or five_frame")
    if training_frame_mode == "five_frame":
        latent = sample_video_latent(
            model, text_hidden_states, frame_count=5, width=resolution, height=resolution,
            steps=denoise_steps, seed=seed, device=device,
            enable_grad_from_step=denoise_steps - draft_k,
        ).to(device=device, dtype=next(vae.parameters()).dtype)
        if checkpoint_vae:
            from torch.utils.checkpoint import checkpoint

            pixels = checkpoint(vae.decode, latent, use_reentrant=False)
        else:
            pixels = vae.decode(latent)
        if pixels.ndim != 5 or pixels.shape[:2] != (1, 3) or pixels.shape[2] < 5:
            raise ValueError(f"Native H3 refinement must decode five RGB frames, got {tuple(pixels.shape)}")
        pixels = ((pixels[:, :, :5].float().clamp(-1, 1) + 1) * 0.5).clamp(0, 1)
        return pixels[0].permute(1, 0, 2, 3)
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


def multiframe_reward_loss(reward, pixels: torch.Tensor, prompt: str, worst_frame_weight: float = 0.0):
    """Score every decoded frame, keeping missed detections in the mean and stop criteria."""
    if not 0.0 <= worst_frame_weight <= 1.0:
        raise ValueError("Worst-frame reward weight must be finite and between 0 and 1")
    values, metrics = [], []
    for frame in pixels:
        values.append(reward(frame.unsqueeze(0).mul(2).sub(1), prompt=prompt).float().mean())
        metrics.append(dict(reward.last_metrics))
    scores = torch.stack(values)
    value = (1 - worst_frame_weight) * scores.mean() + worst_frame_weight * scores.min()
    if not value.requires_grad or not torch.isfinite(value):
        raise RuntimeError("Face reward is not a finite differentiable value")
    reward.last_metrics = {
        key: sum(float(item.get(key, 0)) for item in metrics) / len(metrics)
        for key in ("face_similarity", "face_detected", "pose_similarity")
    }
    reward.last_metrics.update(
        pose_bucket=next((item.get("pose_bucket") for item in metrics if item.get("pose_bucket")), ""),
        reward=float(value.detach()), frame_count=len(metrics),
        worst_frame_reward=float(scores.detach().min()),
    )
    return -value, value.detach()
