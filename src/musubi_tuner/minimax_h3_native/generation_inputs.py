from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
import torch

from musubi_tuner.dataset.media_utils import resize_image_to_bucket
from musubi_tuner.minimax_h3_native.audio_vae import encode_audio_mode
from musubi_tuner.minimax_h3_native.media import (
    H3Record,
    audio_latent_frames,
    load_h3_jsonl_records,
    waveform_samples,
)
from musubi_tuner.minimax_h3_native.packing import H3ReferenceGeometry, H3VideoGeometry, one_frame_condition_role
from musubi_tuner.minimax_h3_native.text_encoder import H3TextVisual
from musubi_tuner.minimax_h3_native.video_vae import VIDEO_VAE_ENCODE_DTYPE, encode_video_condition
from musubi_tuner.minimax_h3_native_cache_latents import PyAVH3MediaDecoder


VIDEO_VAE_SPATIAL_RATIO = 16


def parse_one_frame_options(spec: str) -> tuple[int, tuple[int, ...] | None]:
    """Parse one-frame target/control positions expressed as 24 fps pixel-frame indices."""

    def nonnegative_index(value: str, label: str) -> int:
        try:
            index = int(value)
        except ValueError as error:
            raise ValueError(f"MiniMax-H3 --one_frame {label} must be an integer, got {value!r}") from error
        if index < 0:
            raise ValueError(f"MiniMax-H3 --one_frame {label} must be nonnegative, got {index}")
        return index

    target_index = 0
    control_indices = None
    seen = set()
    for part in spec.split(","):
        key, separator, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value:
            raise ValueError(f"MiniMax-H3 --one_frame options must be key=value, got {part!r}")
        if key not in {"target_index", "control_index"}:
            raise ValueError(
                f"MiniMax-H3 --one_frame has unknown option {key!r} (allowed: target_index, control_index)"
            )
        if key in seen:
            raise ValueError(f"MiniMax-H3 --one_frame has duplicate option {key!r}")
        seen.add(key)
        if key == "target_index":
            target_index = nonnegative_index(value, "target_index")
        else:
            control_indices = tuple(nonnegative_index(item, "control_index") for item in value.split(";"))
    return target_index, control_indices


def dummy_record(prompt: str) -> H3Record:
    return H3Record(
        video_path=Path("."),
        caption=prompt,
        references=(),
        jsonl_line=0,
    )


def load_image_frames(path: str | Path, *, width: int, height: int) -> torch.Tensor:
    """Fit a condition like the training dataset: scale to cover, then center crop."""
    with Image.open(path) as image:
        pixels = resize_image_to_bucket(np.asarray(image.convert("RGB")), (width, height))
    return torch.from_numpy(np.ascontiguousarray(pixels)).unsqueeze(0)


def prepare_pixels(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"MiniMax-H3 condition pixels must be [F,H,W,3], got {tuple(frames.shape)}")
    if frames.dtype == torch.uint8:
        frames = frames.float().div(127.5).sub(1.0)
    else:
        frames = frames.float().mul(2.0).sub(1.0)
    return frames.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def load_generation_record(args) -> H3Record:
    if args.task in {"t2va", "fl2va"}:
        return dummy_record(args.prompt or "")

    records = load_h3_jsonl_records(args.reference_jsonl, "ref2va")
    if args.reference_index >= len(records):
        raise ValueError(f"MiniMax-H3 --reference_index {args.reference_index} is outside {len(records)} JSONL records")
    record = records[args.reference_index]
    if args.prompt is not None:
        record = replace(record, caption=args.prompt)
    return record


def fl_condition_entries(args) -> tuple[tuple[str, str], ...]:
    """Return ordered FL2VA (role, path) entries for video or one-frame generation."""
    first_frame = getattr(args, "first_frame", None)
    last_frame = getattr(args, "last_frame", None)
    condition_images = getattr(args, "condition_image", None) or ()
    if getattr(args, "frame_count", None) != 1:
        if condition_images:
            raise ValueError("MiniMax-H3 condition_image applies only to one-frame FL2VA samples")
        return tuple((role, path) for role, path in (("first", first_frame), ("last", last_frame)) if path)
    if condition_images and (first_frame or last_frame):
        raise ValueError("MiniMax-H3 one-frame FL2VA takes condition images or first/last aliases, not both")
    paths = list(condition_images) if condition_images else [path for path in (first_frame, last_frame) if path]
    return tuple((one_frame_condition_role(index), path) for index, path in enumerate(paths))


def decode_generation_visuals(args, record: H3Record, decoder: PyAVH3MediaDecoder):
    raw_visuals = {}
    text_visuals = {}
    if args.task == "t2va":
        return raw_visuals, text_visuals
    if args.task == "fl2va":
        for role, path in fl_condition_entries(args):
            frames = load_image_frames(path, width=args.width, height=args.height)
            raw_visuals[role] = frames
            text_visuals[role] = H3TextVisual(frames)
        return raw_visuals, text_visuals

    for reference in record.references:
        if reference.type not in {"image", "video"}:
            continue
        frames = decoder.decode_reference_visual(
            reference,
            target_frame_count=args.frame_count,
            target_size=(args.width, args.height),
        )
        raw_visuals[reference.path] = frames
        if reference.type == "image":
            text_visuals[reference.path] = H3TextVisual(frames)
        else:
            sampled = frames[::12]
            text_visuals[reference.path] = H3TextVisual(
                sampled,
                tuple(index / 2.0 for index in range(sampled.shape[0])),
            )
    return raw_visuals, text_visuals


def module_device_dtype(module, fallback_dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return torch.device("cpu"), fallback_dtype


@torch.no_grad()
def encode_visual_conditions(args, record, raw_visuals, video_vae):
    video_device, video_dtype = module_device_dtype(video_vae, VIDEO_VAE_ENCODE_DTYPE)
    visual_latents = []
    visual_geometries = []
    reference_visual_geometries = {}

    def encode_visual(frames):
        latent = encode_video_condition(video_vae, prepare_pixels(frames).to(video_device, video_dtype)).cpu()
        visual_latents.append(latent)
        return H3VideoGeometry(*latent.shape[2:])

    if args.task == "fl2va":
        for role, _ in fl_condition_entries(args):
            visual_geometries.append(encode_visual(raw_visuals[role]))
    elif args.task == "ref2va":
        for index, reference in enumerate(record.references):
            if reference.type in {"image", "video"}:
                reference_visual_geometries[index] = encode_visual(raw_visuals[reference.path])
    return tuple(visual_latents), tuple(visual_geometries), reference_visual_geometries


@torch.no_grad()
def encode_audio_conditions(
    args,
    record,
    decoder,
    audio_vae,
    *,
    reference_video_frame_counts: Mapping[int, int],
):
    audio_device, audio_dtype = module_device_dtype(audio_vae, torch.float32)
    audio_latents = []
    reference_audio_frames = {}
    target_audio_frames = audio_latent_frames(args.frame_count)
    for index, reference in enumerate(record.references):
        if reference.audio is None:
            continue
        if reference.type == "video":
            if index not in reference_video_frame_counts:
                raise ValueError(f"MiniMax-H3 reference video {index:03d} is missing its decoded frame count")
            frame_count = reference_video_frame_counts[index]
            frames = audio_latent_frames(frame_count)
            require_exact = True
        else:
            frames = target_audio_frames
            require_exact = False
        waveform = decoder.decode_audio(
            reference.audio,
            start_sample=0,
            sample_count=waveform_samples(frames),
            require_exact=require_exact,
        )
        latent = encode_audio_mode(audio_vae, waveform.unsqueeze(0).to(audio_device, audio_dtype)).cpu()
        audio_latents.append(latent)
        reference_audio_frames[index] = latent.shape[-1]
    return tuple(audio_latents), reference_audio_frames


def build_reference_geometries(record, visual_geometries, audio_frames):
    references = []
    for index, reference in enumerate(record.references):
        if reference.type == "image":
            references.append(H3ReferenceGeometry("image", video=visual_geometries[index]))
        elif reference.type == "audio":
            references.append(H3ReferenceGeometry("audio", audio_frames=audio_frames[index]))
        else:
            references.append(
                H3ReferenceGeometry(
                    "video",
                    video=visual_geometries[index],
                    audio_frames=audio_frames.get(index, 0),
                )
            )
    return tuple(references)

