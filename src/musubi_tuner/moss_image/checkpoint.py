"""Checkpoint adapters for official and ComfyUI MOSS-Image DiTs.

Model layout adapted from kohya-ss/sd-scripts b8da8698 (Apache-2.0).
Conversion happens as tensors are read; no second full checkpoint is created.
"""
from functools import partial

import torch
from accelerate import init_empty_weights

from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
from musubi_tuner.utils.safetensors_utils import (
    MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks, get_split_weight_filenames,
)
from .models import MOSS_IMAGE_CONFIG, MossImage


def normalize_dit_key(key):
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def split_fused_qkv(key, tensor, dim=3072):
    """Split only main-block Comfy projections; preserve the text refiner's fused QKV."""
    normalized = normalize_dit_key(key)
    # ComfyUI stores per-layer attention configuration as a metadata tensor.
    # It is not part of the trainable transformer state.
    if normalized.endswith(".comfy_attention.config"):
        return [], [] if tensor is not None else None
    parts = normalized.split(".")
    if len(parts) == 5 and parts[0] == "layers" and parts[2:4] == ["proj", "qkv"]:
        suffix = parts[-1]
        if suffix in ("weight", "weight_scale", "comfy_quant"):
            keys = [normalized.replace(".qkv.", f".{p}.") for p in ("q", "k", "v")]
            if tensor is None:
                return keys, None
            if suffix == "comfy_quant":
                return keys, [tensor, tensor, tensor]
            if tensor.shape[0] != 3 * dim:
                raise ValueError(f"{key}: expected {3 * dim} rows, got {tensor.shape[0]}")
            return keys, list(tensor.split(dim, dim=0))
    if normalized != key:
        return [normalized], None if tensor is None else [tensor]
    return None, None


def load_model(path, *, device="cpu", dtype=torch.bfloat16, attn_mode="torch",
               split_attn=False, convrot_int8=False, convrot_bwd_mode="bf16",
               disable_mmap=False, config=None):
    config = dict(MOSS_IMAGE_CONFIG if config is None else config)
    files = get_split_weight_filenames(str(path)) or [str(path)]
    hooks = WeightTransformHooks(split_hook=partial(split_fused_qkv, dim=config["dim"]))
    prequantized = False
    for filename in files:
        with MemoryEfficientSafeOpen(filename, disable_numpy_memmap=disable_mmap) as reader:
            prequantized |= any(k.endswith(".comfy_quant") for k in reader.keys())
    with init_empty_weights():
        model = MossImage(**config, attn_mode=attn_mode, split_attn=split_attn)
    if convrot_int8 or prequantized:
        # Empty targets preserve unquantized layers when the file already carries INT8.
        quantizer = ConvRotInt8Quantizer(
            target_layer_keys=["layers."] if convrot_int8 else [],
            exclude_layer_keys=["norm"], allowed_groupsizes=(256, 64),
        )
        state = quantizer.load_and_quantize(
            files, calc_device="cpu", disable_numpy_memmap=disable_mmap, weight_transform_hooks=hooks,
        )
        apply_convrot_int8_monkey_patch(model, state, bwd_mode=convrot_bwd_mode,
                                       groupsize_map=quantizer.module_groupsizes)
    else:
        state = {}
        for filename in files:
            with MemoryEfficientSafeOpen(filename, disable_numpy_memmap=disable_mmap) as reader:
                adapter = TensorWeightAdapter(hooks, reader)
                for key in adapter.keys():
                    if key in state:
                        raise ValueError(f"Duplicate MOSS checkpoint tensor: {key}")
                    value = adapter.get_tensor(key)
                    if value.is_floating_point() and value.element_size() == 1:
                        raise ValueError("MOSS FP8 checkpoints are not supported; use BF16 or ConvRot INT8.")
                    state[key] = value
    # Derived RoPE frequencies must stay FP32, including after meta initialization.
    state["rope.bands"] = model.rope._default_bands(torch.device("cpu"))
    for key, value in state.items():
        target_dtype = torch.float32 if key.endswith(".scale_weight") or key == "rope.bands" else dtype
        state[key] = value.to(device=device, dtype=target_dtype) if value.is_floating_point() else value.to(device)
    model.requires_grad_(False)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()
