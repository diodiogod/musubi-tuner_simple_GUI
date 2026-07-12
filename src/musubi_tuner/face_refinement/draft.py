"""Musubi-native truncated differentiable sampling for Krea 2 DRaFT."""

from __future__ import annotations

from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from torch.utils.checkpoint import checkpoint

from musubi_tuner.krea2.krea2_sampling import prepare, timesteps


def _velocity(model, image, text, text_mask, negative, negative_mask, timestep, position, negative_position, cfg_scale):
    conditional = model(img=image, context=text, t=timestep, pos=position, mask=text_mask)
    if cfg_scale <= 1 or negative is None:
        return conditional
    with torch.no_grad():
        unconditional = model(
            img=image, context=negative, t=timestep, pos=negative_position, mask=negative_mask
        )
    return unconditional.detach() + cfg_scale * (conditional - unconditional.detach())


def generate_differentiable(
    model,
    vae,
    text,
    text_mask,
    negative,
    negative_mask,
    *,
    resolution: int,
    denoise_steps: int,
    draft_k: int,
    cfg_scale: float,
    seed: int,
    checkpoint_vae: bool = True,
):
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    compression = 2 ** len(vae.temperal_downsample)
    patch = model.config.patch
    latent_size = resolution // compression
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(1, vae.z_dim, latent_size, latent_size, device=device, dtype=dtype, generator=generator)
    text = text.to(device=device, dtype=dtype)
    text_mask = text_mask.to(device=device, dtype=torch.bool)
    negative = None if negative is None else negative.to(device=device, dtype=dtype)
    negative_mask = None if negative_mask is None else negative_mask.to(device=device, dtype=torch.bool)
    image, position, combined_mask = prepare(noise, text.shape[1], patch, text_mask)
    if negative is not None:
        _, negative_position, negative_combined_mask = prepare(noise, negative.shape[1], patch, negative_mask)
    else:
        negative_position = negative_combined_mask = None

    x1 = (256 // (compression * patch)) ** 2
    x2 = (1280 // (compression * patch)) ** 2
    schedule = timesteps(image.shape[1], denoise_steps, x1, x2, y1=0.5, y2=1.15)
    gradient_start = max(0, denoise_steps - draft_k)
    device_type = torch.device(device).type
    for index, (current, previous) in enumerate(zip(schedule[:-1], schedule[1:])):
        timestep = torch.full((1,), current, dtype=image.dtype, device=device)
        delta = previous - current
        context = torch.no_grad() if index < gradient_start else torch.enable_grad()
        with context, torch.autocast(device_type=device_type, dtype=dtype):
            velocity = _velocity(
                model, image, text, combined_mask, negative, negative_combined_mask,
                timestep, position, negative_position, cfg_scale,
            )
            image = image + delta * velocity
        if index < gradient_start:
            image = image.detach()

    latent = rearrange(
        image, "b (h w) (c ph pw) -> b c 1 (h ph) (w pw)",
        ph=patch, pw=patch, h=latent_size // patch,
    )

    def decode(value):
        return vae.decode_to_pixels(value.to(vae.dtype))

    pixels = checkpoint(decode, latent, use_reentrant=False) if checkpoint_vae else decode(latent)
    return pixels.clamp(0, 1)


def reward_loss(reward, pixels: torch.Tensor, prompt: str):
    reward_value = reward(pixels.mul(2).sub(1), prompt).float().mean()
    if not reward_value.requires_grad or not torch.isfinite(reward_value):
        raise RuntimeError("Face reward is not a finite differentiable value")
    return -reward_value, reward_value.detach()


def save_preview(pixels: torch.Tensor, output_dir: str | Path, step: int, prompt: str) -> Path:
    folder = Path(output_dir) / "face_refinement_samples"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step_{step:06d}.png"
    image = pixels[0].detach().float().cpu().clamp(0, 1)
    Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(path)
    path.with_suffix(".txt").write_text(prompt + "\n", encoding="utf-8")
    return path
