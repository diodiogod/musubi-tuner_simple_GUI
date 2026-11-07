"""Official MOSS sigma grid and Euler sampler; adapted from Kohya sd-scripts (Apache-2.0)."""
from typing import Optional
import torch
from tqdm import tqdm
from .models import MossImage

NUM_TRAIN_TIMESTEPS = 1000

def _sigma_grid(count: int, shift: float, total: int) -> torch.Tensor:
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count, dtype=torch.float32)
    return shift * values / (1 + (shift - 1) * values)


def get_sampling_sigmas(
    num_inference_steps: int, shift: float = 5.0, training_shift: float = 1.0, num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    scheduler: str = "official",
) -> torch.Tensor:
    """Return the sampling sigmas [steps + 1] (descending, the last value is 0) on the CPU in float32.

    The number of effective steps can be smaller than `num_inference_steps` because the shifted grid is snapped to the
    training grid and duplicates are removed (same as the official scheduler).
    """
    if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    total = num_train_timesteps
    if scheduler == "simple":
        # Matches ComfyUI simple_scheduler + ModelSamplingDiscreteFlow:
        # https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/samplers.py
        # Select descending entries of its ascending shifted 1/N .. 1 grid.
        grid = torch.arange(1, total + 1, dtype=torch.float32) / total
        grid = shift * grid / (1 + (shift - 1) * grid)
        indices = total - 1 - (torch.arange(num_inference_steps) * (total / num_inference_steps)).long()
        return torch.cat([grid[indices], grid.new_zeros(1)])
    if scheduler != "official":
        raise ValueError(f"Unknown MOSS scheduler: {scheduler}")
    sigmas = _sigma_grid(num_inference_steps, shift, total)
    training = _sigma_grid(total, training_shift, total).flip(0)  # ascending
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def euler_step(latents: torch.Tensor, model_pred: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
    """One Euler step in float32: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v"""
    dt = (sigmas[step_index + 1] - sigmas[step_index]).to(torch.float32)
    return latents.float() + dt * model_pred.float()


def parse_skip_layers(value: Optional[str]) -> tuple[int, ...]:
    """Parse a comma separated list of block indices. None or an empty string disables skipping."""
    if value is None:
        return ()
    value = value.strip()
    if not value:
        return ()
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def generate_latents(
    dit: MossImage,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_embeds: Optional[torch.Tensor],
    negative_mask: Optional[torch.Tensor],
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    flow_shift: float,
    uncond_skip_layers: tuple[int, ...],
    seed: Optional[int],
    device: torch.device,
    show_progress: bool = True,
    sampler: str = "euler",
    scheduler: str = "official",
) -> torch.Tensor:
    """Sample normalized, patchified latents [1, 128, H/16, W/16] (float32).

    The initial noise is drawn on a CPU generator in float32 and CFG / Euler updates are done in float32 like the official
    pipeline. CFG is applied when `guidance_scale != 1` and the negative embeddings are given.
    """
    if height < 16 or width < 16 or height % 16 or width % 16:
        raise ValueError("MOSS image dimensions must be positive multiples of 16.")
    if sampler not in {"euler", "ddim"}:
        raise ValueError(f"Unknown MOSS sampler: {sampler}")
    if any(i < 0 or i >= dit.num_layers for i in uncond_skip_layers):
        raise ValueError("Unconditional skip layer is outside the model block range.")
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    shape = (1, dit.latent_channels, height // 16, width // 16)
    latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)

    positive = prompt_embeds.to(device, dtype=dit.dtype)
    positive_mask = prompt_mask.to(device)
    do_cfg = guidance_scale != 1.0 and negative_embeds is not None
    if do_cfg:
        negative = negative_embeds.to(device, dtype=dit.dtype)
        negative_mask = negative_mask.to(device)

    # ComfyUI's DDIM sampler uses Euler updates; its random-inpaint option has
    # no effect for text-to-image generation without an inpainting mask.
    sigmas = get_sampling_sigmas(steps, flow_shift, scheduler=scheduler).to(device)
    num_steps = len(sigmas) - 1

    with torch.no_grad(), tqdm(total=num_steps, desc="Denoising steps", disable=not show_progress) as pbar:
        for i in range(num_steps):
            timestep = (sigmas[i] * NUM_TRAIN_TIMESTEPS).expand(latents.shape[0])
            x = latents.to(dit.dtype)
            dit.prepare_block_swap_before_forward()
            prediction = dit(x, timestep, positive, positive_mask).float()
            if do_cfg:
                dit.prepare_block_swap_before_forward()
                uncond = dit(x, timestep, negative, negative_mask, skip_layers=uncond_skip_layers).float()
                prediction = uncond + guidance_scale * (prediction - uncond)
            latents = euler_step(latents, prediction, sigmas, i)
            pbar.update()

    return latents
