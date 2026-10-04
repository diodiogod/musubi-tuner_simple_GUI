"""Opt-in cache provenance shared by the GUI's single-pass cache workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import SafetensorError, safe_open

IDENTITY_KEY = "musubi_gui_cache_identity_v1"
_EXECUTION_OPTIONS = {
    "dataset_config",
    "device",
    "batch_size",
    "num_workers",
    "skip_existing",
    "validate_cache",
    "keep_cache",
    "skip_broken",
    "debug_mode",
    "disable_mmap",
    "console_width",
    "console_back",
    "console_num_images",
    "text_encoder_blocks_to_swap",
    "vae_chunk_size",
}
_MODEL_OPTIONS = {"vae", "text_encoder", "t5", "tokenizer"}


def _path_identity(value):
    path = Path(str(value))
    if path.is_file():
        stat = path.stat()
        return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    if path.is_dir():
        # Model/tokenizer folders contain shard indices and configuration as well as weights.
        return [_path_identity(entry) for entry in sorted(path.rglob("*")) if entry.is_file()]
    return str(value)


def cache_recipe(args, kind: str) -> str | None:
    if not getattr(args, "validate_cache", False):
        return None
    options = {
        key: _path_identity(value) if key in _MODEL_OPTIONS and value else value
        for key, value in vars(args).items()
        if key not in _EXECUTION_OPTIONS
    }
    payload = ["cache-v1", kind, options]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _content_identity(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_content_identity(part) for part in value]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    return [str(value.dtype), list(value.shape), hashlib.sha256(memoryview(value).cast("B")).hexdigest()]


def prepare_item_identity(item, recipe: str | None, kind: str) -> None:
    if recipe is None:
        if hasattr(item, "_cache_identity"):
            del item._cache_identity
        return
    payload = {"recipe": recipe, "source": str(item.item_key)}
    if kind == "text":
        payload["caption"] = item.caption
    # Latent retrieval already decodes pixels; hashing those pixels also covers control images
    # whose paths are absent from ItemInfo, preprocessing, audio windows and video crop positions.
    if kind == "latent" or getattr(item, "content", None) is not None:
        for name in ("content", "control_content", "audio_content"):
            payload[name] = _content_identity(getattr(item, name, None))
        for name in (
            "original_size",
            "bucket_size",
            "frame_count",
            "frame_pos",
            "audio_present",
            "fp_1f_clean_indices",
            "fp_1f_target_index",
            "fp_latent_window_size",
        ):
            payload[name] = getattr(item, name, None)
    item._cache_identity = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def identity_metadata(item) -> dict[str, str]:
    identity = getattr(item, "_cache_identity", None)
    return {IDENTITY_KEY: identity} if identity else {}


def cache_identity_matches(item, kind: str) -> bool:
    path = item.latent_cache_path if kind == "latent" else item.text_encoder_output_cache_path
    expected = getattr(item, "_cache_identity", None)
    if not expected:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as cache:
            if (cache.metadata() or {}).get(IDENTITY_KEY) != expected or not list(cache.keys()):
                return False
            # Validate the tensor header too; old, partial and unreadable files rebuild once.
            for key in cache.keys():  # noqa: SIM118 - safetensors handles are not iterable.
                cache.get_slice(key).get_shape()
            return True
    except (OSError, RuntimeError, ValueError, SafetensorError):
        return False
