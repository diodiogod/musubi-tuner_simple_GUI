"""Differentiable MOSS-Image sampling for experimental DRaFT-K refinement."""
import torch
from torch.utils.checkpoint import checkpoint

from musubi_tuner.moss_image.sampling import get_sampling_sigmas


def generate_differentiable_moss(model, vae, text, text_mask, negative, negative_mask, *,
                                 resolution, denoise_steps, draft_k, cfg_scale, seed,
                                 flow_shift=5.0, skip_layers=(22,), checkpoint_vae=True):
    if resolution < 16 or resolution % 16:
        raise ValueError("MOSS face-refinement resolution must be a positive multiple of 16.")
    if not 1 <= draft_k <= denoise_steps:
        raise ValueError("DRaFT K must be between one and the requested denoising steps.")
    if any(layer < 0 or layer >= model.num_layers for layer in skip_layers):
        raise ValueError("A preview skip layer is outside the MOSS transformer.")
    device, dtype = model.device, model.dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, model.latent_channels, resolution // 16, resolution // 16)
    latent = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)
    text, text_mask = text.to(device=device, dtype=dtype), text_mask.to(device)
    if negative is not None:
        negative, negative_mask = negative.to(device=device, dtype=dtype), negative_mask.to(device)
    sigmas = get_sampling_sigmas(denoise_steps, flow_shift).to(device)
    effective_steps = len(sigmas) - 1
    gradient_start = max(0, effective_steps - draft_k)
    for step in range(effective_steps):
        gradient_step = step >= gradient_start
        time = (sigmas[step] * 1000).reshape(1)
        if cfg_scale != 1 and negative is not None:
            with torch.no_grad():
                model.prepare_block_swap_before_forward()
                uncond = model(latent.to(dtype), time, negative, negative_mask, skip_layers=skip_layers).float()
        with torch.set_grad_enabled(gradient_step):
            model.prepare_block_swap_before_forward()
            cond = model(latent.to(dtype), time, text, text_mask).float()
            velocity = uncond + cfg_scale * (cond - uncond) if cfg_scale != 1 and negative is not None else cond
            latent = latent + (sigmas[step + 1] - sigmas[step]) * velocity
        if not gradient_step:
            latent = latent.detach()

    def decode(value):
        with torch.autocast(device_type=device.type, enabled=False):
            return vae.decode(value.float())

    pixels = checkpoint(decode, latent, use_reentrant=False) if checkpoint_vae else decode(latent)
    return pixels.float().add(1).div(2).clamp(0, 1)
