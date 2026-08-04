from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from pathlib import Path
from typing import TypeVar

from accelerate import init_empty_weights
import torch
import torch.nn as nn

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def inspect_convrot_int8_checkpoint(
    files: Iterable[str | Path],
    *,
    key_prefixes: tuple[str, ...] = (),
) -> tuple[dict[str, torch.Size], dict[str, tuple[torch.Size, torch.dtype]]]:
    """Validate Comfy's pre-quantized ConvRot contract without loading base weights.

    Returns the runtime scale-buffer shapes keyed by module path and a compact
    key manifest used to infer the pruned architecture.  Tensor access is
    memory-mapped, so inspecting a ~21 GB checkpoint does not materialize it.
    """
    files = [Path(path).resolve() for path in files]
    manifest: dict[str, tuple[torch.Size, torch.dtype]] = {}
    markers: dict[str, dict] = {}

    for path in files:
        with MemoryEfficientSafeOpen(str(path)) as handle:
            for raw_key in handle.keys():
                key = _normalize_checkpoint_key(raw_key, key_prefixes)
                if key in manifest:
                    raise ValueError(f"Duplicate MiniMax-H3 checkpoint key {key!r} in {path}")
                tensor = handle.get_tensor(raw_key, device=torch.device("cpu"), dtype=None)
                manifest[key] = (tensor.shape, tensor.dtype)
                if key.endswith(".comfy_quant"):
                    try:
                        payload = bytes(tensor.reshape(-1).tolist()).decode("utf-8").rstrip("\x00")
                        marker = json.loads(payload)
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
                        raise ValueError(f"Invalid ConvRot metadata tensor {key!r} in {path}") from error
                    markers[key.removesuffix(".comfy_quant")] = marker

    scale_bases = {key.removesuffix(".weight_scale") for key in manifest if key.endswith(".weight_scale")}
    marker_bases = set(markers)
    if not marker_bases:
        raise ValueError("MiniMax-H3 checkpoint contains no ConvRot INT8 metadata tensors")
    if scale_bases != marker_bases:
        raise ValueError(
            "MiniMax-H3 ConvRot marker/scale mismatch: "
            f"missing_markers={sorted(scale_bases - marker_bases)[:20]}, "
            f"missing_scales={sorted(marker_bases - scale_bases)[:20]}"
        )

    scale_shapes: dict[str, torch.Size] = {}
    for base in sorted(marker_bases):
        marker = markers[base]
        expected_marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        if marker != expected_marker:
            raise ValueError(f"Unsupported MiniMax-H3 quantization metadata for {base}: {marker!r}")
        weight_key = f"{base}.weight"
        scale_key = f"{base}.weight_scale"
        if weight_key not in manifest:
            raise ValueError(f"MiniMax-H3 ConvRot module {base} is missing {weight_key}")
        weight_shape, weight_dtype = manifest[weight_key]
        scale_shape, scale_dtype = manifest[scale_key]
        if weight_dtype != torch.int8 or len(weight_shape) != 2:
            raise ValueError(f"MiniMax-H3 ConvRot weight {weight_key} must be rank-2 INT8")
        if scale_dtype != torch.float32 or tuple(scale_shape) != (weight_shape[0], 1):
            raise ValueError(
                f"MiniMax-H3 ConvRot scale {scale_key} must be FP32 [{weight_shape[0]},1], "
                f"got {scale_dtype} {tuple(scale_shape)}"
            )
        scale_shapes[base] = scale_shape
    return scale_shapes, manifest


def resolve_safetensors_files(path: str | Path) -> list[Path]:
    path = Path(path).resolve()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Expected a safetensors checkpoint, got {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Checkpoint path does not exist: {path}")

    index_files = sorted(path.glob("*.safetensors.index.json"))
    if len(index_files) > 1:
        raise ValueError(f"Multiple safetensors indexes found in {path}: {index_files}")
    if index_files:
        try:
            index = json.loads(index_files[0].read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid safetensors index: {index_files[0]}") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors index has no weights: {index_files[0]}")

        files = []
        for filename in sorted(set(weight_map.values())):
            if not isinstance(filename, str):
                raise ValueError(f"Invalid shard filename in {index_files[0]}: {filename!r}")
            shard = (path / filename).resolve()
            if not shard.is_relative_to(path) or not shard.is_file():
                raise ValueError(f"Missing or unsafe safetensors shard in {index_files[0]}: {filename}")
            files.append(shard)
        return files

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors checkpoint found in {path}")
    if len(files) > 1:
        raise ValueError(f"Multiple safetensors files without an index found in {path}: {files}")
    return [files[0].resolve()]


def _normalize_checkpoint_key(key: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def load_safetensors_metadata(files: Iterable[str | Path]) -> dict[str, str]:
    merged = {}
    for file in files:
        path = Path(file).resolve()
        with MemoryEfficientSafeOpen(str(path)) as handle:
            metadata = handle.metadata() or {}
        for key, value in metadata.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"Conflicting MiniMax-H3 checkpoint metadata {key!r} in {path}")
            merged[key] = value
    return merged


def _format_key_mismatch(
    missing: Iterable[str],
    unexpected: Iterable[str],
    shape_mismatches: Iterable[str],
    dtype_mismatches: Iterable[str],
) -> str:
    missing = sorted(missing)
    unexpected = sorted(unexpected)
    shape_mismatches = sorted(shape_mismatches)
    dtype_mismatches = sorted(dtype_mismatches)
    return (
        "MiniMax-H3 checkpoint key mismatch: "
        f"missing={missing[:20]}, unexpected={unexpected[:20]}, shape_mismatches={shape_mismatches[:20]}, "
        f"dtype_mismatches={dtype_mismatches[:20]}"
    )


def load_safetensors_module(
    factory: Callable[[], ModuleT],
    files: Iterable[str | Path],
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
    key_prefixes: tuple[str, ...] = (),
    key_transform: Callable[[str], str] | None = None,
    ignore_key: Callable[[str], bool] | None = None,
    strict_dtype: bool = False,
    cast_to_expected_dtype: bool = False,
    allow_quantized: bool = False,
    disable_mmap: bool = False,
) -> ModuleT:
    files = [Path(path).resolve() for path in files]
    if not files:
        raise ValueError("MiniMax-H3 checkpoint file list is empty")

    with init_empty_weights():
        model = factory()

    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    seen = set()
    unexpected = set()
    shape_mismatches = []
    dtype_mismatches = []

    for path in files:
        shard = {}
        with MemoryEfficientSafeOpen(str(path), disable_numpy_memmap=disable_mmap) as handle:
            raw_keys = handle.keys()
            if not allow_quantized and any(key.endswith((".weight_scale", ".comfy_quant")) for key in raw_keys):
                raise ValueError(
                    "Quantized MiniMax-H3 artifacts require the explicit pruned ConvRot INT8 loader; "
                    "generic quantized loading remains deferred to R2"
                )
            for raw_key in raw_keys:
                key = _normalize_checkpoint_key(raw_key, key_prefixes)
                if ignore_key is not None and ignore_key(key):
                    continue
                if key_transform is not None:
                    key = key_transform(key)
                if key in seen:
                    raise ValueError(f"Duplicate MiniMax-H3 checkpoint key {key!r} in {path}")
                seen.add(key)
                if key not in expected_keys:
                    unexpected.add(key)
                    continue
                tensor = handle.get_tensor(raw_key, device=torch.device("cpu"), dtype=None)
                if tensor.shape != expected_state[key].shape:
                    shape_mismatches.append(f"{key}: expected {tuple(expected_state[key].shape)}, got {tuple(tensor.shape)}")
                    continue
                if strict_dtype and tensor.dtype != expected_state[key].dtype:
                    dtype_mismatches.append(f"{key}: expected {expected_state[key].dtype}, got {tensor.dtype}")
                    continue
                if cast_to_expected_dtype and tensor.is_floating_point():
                    tensor = tensor.to(dtype=expected_state[key].dtype)
                elif dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                shard[key] = tensor
        if shard:
            model.load_state_dict(shard, strict=False, assign=True)

    missing = expected_keys - seen
    if missing or unexpected or shape_mismatches or dtype_mismatches:
        raise ValueError(_format_key_mismatch(missing, unexpected, shape_mismatches, dtype_mismatches))

    model.to(device=torch.device(device))
    model.eval()
    return model
