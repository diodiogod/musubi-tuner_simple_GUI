"""ConvRot INT8 optimization for frozen base weights during LoRA training.

Mirrors the structure of fp8_optimization_utils.py: quantize target Linear weights at
load time, store the INT8 weight (in the rotated basis) under the original ``.weight``
key and the per-channel scale under a sibling ``.scale_weight`` key, then monkey-patch
the matching ``nn.Linear`` modules' ``forward``.

Unlike the fp8 path, the forward goes through a custom ``torch.autograd.Function``:
the fused Triton kernel (rotation + dynamic row-wise INT8 quantization + INT8 GEMM
with dequantization epilogue) has no autograd support, and the base weight is frozen
so only grad_x is needed in backward: grad_x = rotate(g @ W_rot).

Keeping the module an ``nn.Linear`` (patched forward, INT8 ``.weight``) is load-bearing:
LoRA targets modules by class name "Linear", block swap streams ``module.weight.data``
of Linear-named modules, and compile exclusion also keys on the class name.
"""

import os
import types
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from musubi_tuner.modules.convrot_int8_kernels import (
    HAS_TRITON,
    _build_hadamard,
    _rotate_activation,
    int8_linear,
    quantize_int8_convrot_weight,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONVROT_GROUPSIZE = 256


def quantize_weight_convrot(key: str, tensor: torch.Tensor, groupsize: int = CONVROT_GROUPSIZE):
    """Quantize a single weight tensor with ConvRot INT8, or return None if not applicable.

    The quantization is always done with the eager implementation: it is deterministic
    and independent of triton availability, so the resulting state dict is identical
    across environments.

    Returns:
        (quantized int8 [N, K] in rotated basis, scale float32 [N, 1]), or None if the
        tensor is not a 2D weight or its in_features is not divisible by groupsize.
    """
    if tensor.ndim != 2:
        logger.info(f"Skipping ConvRot INT8 for {key}: not a 2D weight (ndim={tensor.ndim})")
        return None
    if tensor.shape[1] % groupsize != 0:
        logger.info(f"Skipping ConvRot INT8 for {key}: in_features {tensor.shape[1]} not divisible by {groupsize}")
        return None
    return quantize_int8_convrot_weight(tensor, groupsize)


class ConvRotInt8Quantizer:
    """Strategy object that streams safetensors files and quantizes target weights.

    Passed as ``quantizer`` to ``load_safetensors_with_lora_and_fp8``; carries its own
    streaming loader so the fp8 path stays untouched.
    """

    def __init__(
        self,
        target_layer_keys: Optional[List[str]] = None,
        exclude_layer_keys: Optional[List[str]] = None,
        groupsize: int = CONVROT_GROUPSIZE,
    ):
        self.target_layer_keys = target_layer_keys
        self.exclude_layer_keys = exclude_layer_keys
        self.groupsize = groupsize

    def is_target_key(self, key: str) -> bool:
        is_target = (
            self.target_layer_keys is None or any(pattern in key for pattern in self.target_layer_keys)
        ) and key.endswith(".weight")
        is_excluded = self.exclude_layer_keys is not None and any(pattern in key for pattern in self.exclude_layer_keys)
        return is_target and not is_excluded

    def load_and_quantize(
        self,
        model_files: List[str],
        calc_device: Union[str, torch.device, None],
        move_to_device: bool = False,
        weight_hook: Optional[callable] = None,
        disable_numpy_memmap: bool = False,
        weight_transform_hooks: Optional[WeightTransformHooks] = None,
    ) -> dict:
        """Load state dict from safetensors files, quantizing target weights to ConvRot INT8.

        Same streaming contract as load_safetensors_with_fp8_optimization: the LoRA merge
        weight_hook runs on the raw (bf16) weight before quantization.
        """
        optimized_count = 0
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f

                keys = f.keys()
                for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                    value = f.get_tensor(key)
                    original_device = value.device  # usually cpu

                    if weight_hook is not None:
                        value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                    if not self.is_target_key(key):
                        target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                        state_dict[key] = value.to(target_device)
                        continue

                    if calc_device is not None:
                        value = value.to(calc_device)

                    if value.dtype.itemsize == 1:
                        raise ValueError(
                            f"Layer {key} is already in {value.dtype} format. Loading pre-quantized weights with"
                            " --convrot_int8 is not supported yet. Please use fp16/bf16/float32 model weights."
                            + f" / レイヤー {key} は既に{value.dtype}形式です。事前量子化済み重みの --convrot_int8 での"
                            "読み込みは未対応です。FP16/BF16/Float32のモデル重みを使用してください。"
                        )

                    result = quantize_weight_convrot(key, value, self.groupsize)
                    if result is None:
                        # leave the layer unquantized (bf16)
                        if not move_to_device:
                            value = value.to(original_device)
                        state_dict[key] = value
                        continue
                    quantized_weight, scale_tensor = result

                    scale_key = key.replace(".weight", ".scale_weight")
                    assert key != scale_key, "weight key and scale key must be different"

                    if not move_to_device:
                        quantized_weight = quantized_weight.to(original_device)

                    # scale stays float32 [N, 1]: the Triton epilogue and the backward want fp32,
                    # and the shape maps 1:1 to ComfyUI's `weight_scale`
                    state_dict[key] = quantized_weight
                    state_dict[scale_key] = scale_tensor.to(device=quantized_weight.device)

                    optimized_count += 1

                    if calc_device is not None and optimized_count % 10 == 0:
                        clean_memory_on_device(calc_device)

        logger.info(f"Number of ConvRot INT8 optimized Linear layers: {optimized_count}")
        return state_dict


class ConvRotInt8LinearFn(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, wq, w_scale, bias, groupsize, bwd_mode):
        # x: [..., K] bf16/fp16, wq: [N, K] int8 (rotated basis), w_scale: [N, 1] fp32
        # F.linear casts its inputs to the autocast dtype under autocast; the fused kernel
        # bypasses F.linear, so replicate that here. In K2 the fp32 modulation adds promote
        # the activations to fp32, and downstream flash-attn only accepts fp16/bf16.
        if torch.is_autocast_enabled(x.device.type):
            cast_dtype = torch.get_autocast_dtype(x.device.type)
            x = x.to(cast_dtype)
            if bias is not None:
                bias = bias.to(cast_dtype)
        if HAS_TRITON and x.is_cuda:
            out = int8_linear(x, wq, w_scale.reshape(-1), bias, x.dtype, True, groupsize)
        else:
            # eager fallback: rotation + transient dequantized matmul, no activation quantization
            h = _build_hadamard(groupsize, device=x.device, dtype=x.dtype)
            x_rot = _rotate_activation(x, h, groupsize)
            w_rot = wq.to(x.dtype) * w_scale.reshape(-1, 1).to(x.dtype)
            out = F.linear(x_rot, w_rot, bias)
        # wq/w_scale are live buffers, so saving them adds no activation memory; x is not
        # saved (base is frozen, no grad_weight needed)
        ctx.save_for_backward(wq, w_scale)
        ctx.groupsize = groupsize
        ctx.bwd_mode = bwd_mode
        ctx.bias_needs_grad = bias is not None and bias.requires_grad
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        wq, w_scale = ctx.saved_tensors
        gs = ctx.groupsize
        g2d = grad_out.reshape(-1, grad_out.shape[-1])  # [M, N]

        grad_x = None
        if ctx.needs_input_grad[0]:
            # grad_x = g @ W = g @ (W_rot R) = rotate(g @ W_rot), R = block-diag Hadamard
            if ctx.bwd_mode == "int8":
                # fold per-channel weight scale into g, then reuse the fused Triton GEMM
                # (row-wise quant of g + int8 GEMM + dequant epilogue in one pipeline).
                # transient int8 transpose of wq: [K, N], ~1 byte/param, freed after mm
                g_scaled = g2d * w_scale.reshape(1, -1).to(g2d.dtype)
                one = torch.ones(1, device=g2d.device, dtype=torch.float32)
                gx_rot = int8_linear(g_scaled, wq.t().contiguous(), one, None, grad_out.dtype, False, gs)
            else:
                # transient bf16 dequant of the rotated weight (stays in rotated basis)
                w_rot = wq.to(grad_out.dtype) * w_scale.reshape(-1, 1).to(grad_out.dtype)
                gx_rot = g2d @ w_rot  # [M, K]
            h = _build_hadamard(gs, device=gx_rot.device, dtype=gx_rot.dtype)
            grad_x = _rotate_activation(gx_rot, h, gs).reshape(*grad_out.shape[:-1], wq.shape[1])

        grad_bias = g2d.sum(dim=0) if ctx.bias_needs_grad else None
        return grad_x, None, None, grad_bias, None, None


def convrot_int8_linear_forward_patch(self: nn.Linear, x):
    return ConvRotInt8LinearFn.apply(
        x, self.weight, self.scale_weight, self.bias, self._convrot_groupsize, self._convrot_bwd_mode
    )


def patch_convrot_int8_modules(
    model: nn.Module,
    scale_shapes: dict[str, torch.Size | tuple[int, ...]],
    *,
    bwd_mode: str = "bf16",
    groupsize: int = CONVROT_GROUPSIZE,
) -> nn.Module:
    """Convert selected ``nn.Linear`` modules into frozen ConvRot INT8 linears.

    The replacement keeps the original module class so existing LoRA targeting
    and block-streaming code continue to recognize it.  It is also safe on the
    meta device, which lets a pre-quantized checkpoint be assigned directly
    without ever constructing a full-precision base weight.
    """
    """
    ``bf16`` backward is the portable correctness path. ``int8`` backward is
    opt-in and requires Triton because it quantizes the activation gradient.
    """
    if bwd_mode not in ("bf16", "int8"):
        raise ValueError(f"Unsupported ConvRot INT8 backward mode: {bwd_mode}")
    if bwd_mode == "int8" and not HAS_TRITON:
        raise ValueError("ConvRot INT8 backward mode 'int8' requires triton. Install triton (triton-windows on Windows).")
    if not HAS_TRITON:
        logger.warning(
            "triton is not available: ConvRot INT8 falls back to transient dequantization in forward."
            " Weight VRAM is still reduced, but there is no speedup. Install triton (triton-windows on Windows)"
            " for the fused INT8 kernels."
        )

    patched_module_paths = set(scale_shapes)
    patched_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in patched_module_paths:
            if module.weight.shape[1] % groupsize:
                raise ValueError(
                    f"ConvRot INT8 module {name} has in_features={module.weight.shape[1]}, not divisible by {groupsize}"
                )
            expected_scale_shape = (module.weight.shape[0], 1)
            if tuple(scale_shapes[name]) != expected_scale_shape:
                raise ValueError(
                    f"ConvRot INT8 scale for {name} expected {expected_scale_shape}, got {tuple(scale_shapes[name])}"
                )
            weight = torch.empty(module.weight.shape, dtype=torch.int8, device=module.weight.device)
            module.weight = nn.Parameter(weight, requires_grad=False)
            module.register_buffer(
                "scale_weight",
                torch.empty(expected_scale_shape, dtype=torch.float32, device=weight.device),
            )
            module._convrot_groupsize = groupsize
            module._convrot_bwd_mode = bwd_mode
            module.forward = types.MethodType(convrot_int8_linear_forward_patch, module)

            patched_count += 1

    logger.info(f"Number of ConvRot INT8 monkey-patched Linear layers: {patched_count}")
    missing = patched_module_paths - {name for name, module in model.named_modules() if hasattr(module, "scale_weight")}
    if missing:
        raise ValueError(f"ConvRot INT8 checkpoint references unknown Linear modules: {sorted(missing)[:20]}")
    return model


def apply_convrot_int8_monkey_patch(model, optimized_state_dict, bwd_mode: str = "bf16", groupsize: int = CONVROT_GROUPSIZE):
    """Patch a model from an on-the-fly quantizer state dictionary."""
    scale_shapes = {
        key.rsplit(".scale_weight", 1)[0]: value.shape
        for key, value in optimized_state_dict.items()
        if key.endswith(".scale_weight")
    }
    return patch_convrot_int8_modules(model, scale_shapes, bwd_mode=bwd_mode, groupsize=groupsize)
