# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted for the downstream compact ConvRot preview path from the native
# MiniMax-H3 joint sampler in kohya-ss/musubi-tuner PR #1018.

"""Native short-video sampling helpers for compact MiniMax-H3 previews."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import av
import torch

from musubi_tuner.minimax_h3.media import audio_latent_frames, video_latent_frames
from musubi_tuner.minimax_h3.packing import H3VideoGeometry, build_h3_layout


VIDEO_VAE_SPATIAL_RATIO = 16


def build_joint_sigma_schedule(
    steps: int,
    *,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError(f"MiniMax-H3 sampling steps must be positive, got {steps}")
    for value, label in ((video_shift, "video"), (audio_shift, "audio")):
        if not 0.01 <= float(value) <= 100.0:
            raise ValueError(f"MiniMax-H3 {label} shift must be in [0.01,100.0], got {value}")
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64, device=device)

    def shifted(value: float) -> torch.Tensor:
        return value * base / (1.0 + (value - 1.0) * base)

    return shifted(float(video_shift)), shifted(float(audio_shift))


def _map_shifted_sigma(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def _shifted_sigma_slope(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    numerator = to_shift * (1.0 + (from_shift - 1.0) * base).square()
    denominator = from_shift * (1.0 + (to_shift - 1.0) * base).square()
    return numerator / denominator


def _res_multistep_update(
    sample: torch.Tensor,
    denoised: torch.Tensor,
    *,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    sigma_previous: torch.Tensor | None,
    old_denoised: torch.Tensor | None,
) -> torch.Tensor:
    """Deterministic RES multistep update used by ComfyUI's H3 workflow."""
    if float(sigma_next) == 0.0 or old_denoised is None:
        derivative = (sample - denoised) / sigma.to(sample)
        return sample + derivative * (sigma_next - sigma).to(sample)

    t = -sigma.log()
    t_next = -sigma_next.log()
    t_previous = -sigma_previous.log()
    h = t_next - t
    c2 = (t_previous - t) / h
    phi1 = torch.expm1(-h) / -h
    phi2 = (phi1 - 1.0) / -h
    b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
    b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
    return torch.exp(-h).to(sample) * sample + h.to(sample) * (
        b1.to(sample) * denoised + b2.to(sample) * old_denoised
    )


def initialize_joint_noise(
    *,
    frame_count: int,
    width: int,
    height: int,
    seed: int,
    device: torch.device | str,
    video_dtype: torch.dtype = torch.float32,
    audio_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError(f"MiniMax-H3 video size must be positive and divisible by 32, got {width}x{height}")
    video_frames = video_latent_frames(int(frame_count))
    audio_frames = audio_latent_frames(int(frame_count))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    video = torch.randn(
        (1, 24, video_frames, height // VIDEO_VAE_SPATIAL_RATIO, width // VIDEO_VAE_SPATIAL_RATIO),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=video_dtype)
    audio = torch.randn(
        (1, 32, 2, audio_frames),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=audio_dtype)
    return video, audio


@torch.no_grad()
def sample_video_latent(
    transformer,
    text_hidden_states: torch.Tensor,
    *,
    frame_count: int,
    width: int,
    height: int,
    steps: int,
    seed: int,
    device: torch.device | str,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    step_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    device = torch.device(device)
    text = text_hidden_states.to(device=device, dtype=torch.bfloat16)
    if text.ndim == 2:
        text = text.unsqueeze(0)
    if text.ndim != 3 or text.shape[0] != 1 or text.shape[-1] != 5120:
        raise ValueError(f"MiniMax-H3 video text states must be [1,L,5120], got {tuple(text.shape)}")
    video, audio = initialize_joint_noise(
        frame_count=frame_count,
        width=width,
        height=height,
        seed=seed,
        device=device,
    )
    layout = build_h3_layout(
        task="t2va",
        text_length=text.shape[1],
        target_video=H3VideoGeometry(video.shape[2], video.shape[3], video.shape[4]),
        target_audio_frames=audio.shape[-1],
    )
    token_tags = torch.ones((1, text.shape[1]), dtype=torch.int64, device=device)
    video_sigmas, _audio_sigmas = build_joint_sigma_schedule(
        steps,
        video_shift=video_shift,
        audio_shift=audio_shift,
        device=device,
    )
    old_video_denoised = None
    old_audio_denoised = None
    for index in range(steps):
        sigma_video = video_sigmas[index].to(torch.float32)
        sigma_audio = _map_shifted_sigma(sigma_video, float(video_shift), float(audio_shift))
        prediction = transformer(
            video_latents=video,
            audio_latents=audio,
            text_hidden_states=text,
            text_token_tags=token_tags,
            layout=layout,
            model_t_video=1.0 - sigma_video,
            model_t_audio=1.0 - sigma_audio,
        )
        if prediction.video.shape != video.shape or prediction.audio.shape != audio.shape:
            raise ValueError("MiniMax-H3 transformer predictions do not match the target latent shapes")
        audio_slope = _shifted_sigma_slope(sigma_video, float(video_shift), float(audio_shift))
        video_denoised = video + sigma_video.to(video) * prediction.video
        audio_denoised = audio + (sigma_video * audio_slope).to(audio) * prediction.audio
        sigma_previous = video_sigmas[index - 1] if index else None
        video_next = _res_multistep_update(
            video,
            video_denoised,
            sigma=sigma_video,
            sigma_next=video_sigmas[index + 1],
            sigma_previous=sigma_previous,
            old_denoised=old_video_denoised,
        )
        audio_next = _res_multistep_update(
            audio,
            audio_denoised,
            sigma=sigma_video,
            sigma_next=video_sigmas[index + 1],
            sigma_previous=sigma_previous,
            old_denoised=old_audio_denoised,
        )
        old_video_denoised, old_audio_denoised = video_denoised, audio_denoised
        video, audio = video_next, audio_next
        if step_callback is not None:
            step_callback(index + 1, steps)
    return video


@torch.no_grad()
def decode_video_latent(video_vae, latent: torch.Tensor, *, frame_count: int) -> torch.Tensor:
    pixels = video_vae.decode(latent)
    if pixels.ndim != 5 or pixels.shape[:2] != (1, 3):
        raise ValueError(f"MiniMax-H3 video VAE must decode [1,3,F,H,W], got {tuple(pixels.shape)}")
    pixels = pixels[0, :, :frame_count].detach().cpu().float().clamp(-1.0, 1.0)
    return ((pixels + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).contiguous()


def write_silent_video(video: torch.Tensor, output: str | Path, *, fps: int = 24) -> Path:
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != torch.uint8:
        raise ValueError(f"MiniMax-H3 output video must be uint8 [F,H,W,3], got {tuple(video.shape)} {video.dtype}")
    output = Path(output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Compact MiniMax-H3 video previews must use .mp4 output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = video.shape[2]
        stream.height = video.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, fps)
        for index, pixels in enumerate(video):
            frame = av.VideoFrame.from_ndarray(pixels.numpy(), format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output
