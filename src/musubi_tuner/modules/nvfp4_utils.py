# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org
# SPDX-License-Identifier: Apache-2.0

"""Standalone packed-NVFP4 inference helpers for MiniMax-H3 text caching.

The scale addressing follows ComfyUI/comfy-kitchen's cuBLAS 128x4 blocked layout,
but the decoder deliberately uses ordinary bit operations so it also works on Ada
GPUs without native SM100 FP4 conversion instructions.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
logger = logging.getLogger(__name__)
_TRITON_FAILURE_REPORTED = False


def from_blocked(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Invert ComfyUI's 128x4 scale swizzle into logical row-major order."""

    nrb = -(-rows // 128)
    ncb = -(-cols // 4)
    x = blocked.reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(nrb, ncb, 128, 4).permute(0, 2, 1, 3).reshape(nrb * 128, ncb * 4)
    return x[:rows, :cols].contiguous()


def dequantize_nvfp4_eager(
    packed: torch.Tensor,
    block_scale_fp8: torch.Tensor,
    global_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference packed NVFP4 decoder used by CPU tests and the no-Triton fallback."""

    out_features, packed_in_features = packed.shape
    in_features = packed_in_features * 2
    low = (packed & 0x0F).to(torch.long)
    high = (packed >> 4).to(torch.long)
    codes = torch.stack([high, low], dim=-1).reshape(out_features, in_features)
    sign = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    values = sign * _E2M1_MAG.to(codes.device)[codes & 0x7]
    scales = from_blocked(block_scale_fp8.to(torch.float32).reshape(-1, 32, 16), out_features, in_features // 16)
    scales = scales.repeat_interleave(16, dim=1)
    return (values * scales * global_scale.to(device=values.device, dtype=torch.float32)).to(output_dtype)


if HAS_TRITON:

    @triton.jit
    def _dequantize_nvfp4_ada_kernel(
        packed_ptr,
        scale_ptr,
        global_scale_ptr,
        output_ptr,
        total_elements,
        in_features: tl.constexpr,
        packed_in_features: tl.constexpr,
        scale_cols: tl.constexpr,
        n_col_blocks: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total_elements
        rows = offsets // in_features
        cols = offsets - rows * in_features

        packed_offsets = rows * packed_in_features + cols // 2
        packed_values = tl.load(packed_ptr + packed_offsets, mask=mask, other=0).to(tl.uint8)
        high_nibble = packed_values >> 4
        low_nibble = packed_values & 0x0F
        codes = tl.where((cols & 1) == 0, high_nibble, low_nibble).to(tl.int32)

        magnitude_code = codes & 0x7
        magnitude = tl.where(
            magnitude_code == 0,
            0.0,
            tl.where(
                magnitude_code == 1,
                0.5,
                tl.where(
                    magnitude_code == 2,
                    1.0,
                    tl.where(
                        magnitude_code == 3,
                        1.5,
                        tl.where(magnitude_code == 4, 2.0, tl.where(magnitude_code == 5, 3.0, tl.where(magnitude_code == 6, 4.0, 6.0))),
                    ),
                ),
            ),
        )
        values = tl.where((codes & 0x8) != 0, -magnitude, magnitude)

        scale_columns = cols // 16
        row_blocks = rows // 128
        column_blocks = scale_columns // 4
        rows_in_block = rows % 128
        sub_blocks = rows_in_block // 32
        fine_rows = rows_in_block % 32
        columns_in_block = scale_columns % 4
        combined_blocks = row_blocks * n_col_blocks + column_blocks
        scale_offsets = combined_blocks * 512 + fine_rows * 16 + sub_blocks * 4 + columns_in_block
        scales = tl.load(scale_ptr + scale_offsets, mask=mask & (scale_columns < scale_cols), other=1.0).to(tl.float32)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        tl.store(output_ptr + offsets, values.to(tl.float32) * scales * global_scale, mask=mask)


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale_fp8: torch.Tensor,
    global_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode one packed weight matrix without permanently expanding the model."""

    if not (HAS_TRITON and packed.is_cuda):
        return dequantize_nvfp4_eager(packed, block_scale_fp8, global_scale, output_dtype)
    if packed.ndim != 2:
        raise ValueError(f"Packed NVFP4 weight must be two-dimensional, got {tuple(packed.shape)}")
    out_features, packed_in_features = packed.shape
    in_features = packed_in_features * 2
    if in_features % 16:
        raise ValueError(f"NVFP4 in_features must be divisible by 16, got {in_features}")
    output = torch.empty((out_features, in_features), device=packed.device, dtype=output_dtype)
    total_elements = output.numel()
    scale_cols = in_features // 16
    n_col_blocks = -(-scale_cols // 4)
    grid = (triton.cdiv(total_elements, 256),)
    try:
        _dequantize_nvfp4_ada_kernel[grid](
            packed,
            block_scale_fp8,
            global_scale,
            output,
            total_elements,
            in_features=in_features,
            packed_in_features=packed_in_features,
            scale_cols=scale_cols,
            n_col_blocks=n_col_blocks,
            block_size=256,
        )
        return output
    except Exception:
        global _TRITON_FAILURE_REPORTED
        if not _TRITON_FAILURE_REPORTED:
            logger.exception("Packed NVFP4 Triton decoder failed; using the slower eager decoder")
            _TRITON_FAILURE_REPORTED = True
        del output
        return dequantize_nvfp4_eager(packed, block_scale_fp8, global_scale, output_dtype)


class PackedNVFP4Linear(nn.Module):
    """Frozen Linear that keeps Comfy NVFP4 weights packed between forwards."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, device: str | torch.device = "meta"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features // 2), dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.register_buffer("weight_scale", None)
        self.register_buffer("weight_scale_2", None)
        self.register_buffer("pre_quant_scale", None)
        self.compute_dtype = torch.bfloat16

    def load_quantized(
        self,
        packed: torch.Tensor,
        block_scale: torch.Tensor,
        global_scale: torch.Tensor,
        device: torch.device,
        compute_dtype: torch.dtype,
        pre_quant_scale: torch.Tensor | None = None,
    ) -> None:
        expected = (self.out_features, self.in_features // 2)
        if tuple(packed.shape) != expected or packed.dtype != torch.uint8:
            raise ValueError(f"NVFP4 packed weight must be uint8 {expected}, got {packed.dtype} {tuple(packed.shape)}")
        self.weight = nn.Parameter(packed.to(device=device), requires_grad=False)
        self.weight_scale = block_scale.to(device=device)
        self.weight_scale_2 = global_scale.to(device=device, dtype=torch.float32)
        self.pre_quant_scale = None if pre_quant_scale is None else pre_quant_scale.to(device=device, dtype=compute_dtype)
        self.compute_dtype = compute_dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.pre_quant_scale is not None:
            inputs = inputs * self.pre_quant_scale.to(dtype=inputs.dtype)
        weight = dequantize_nvfp4(self.weight, self.weight_scale, self.weight_scale_2, inputs.dtype)
        return F.linear(inputs, weight, self.bias)


class PackedInt8Embedding(nn.Module):
    """Frozen INT8 embedding that dequantizes only selected token rows."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device: str | torch.device = "meta"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.register_buffer("weight_scale", None)
        self.compute_dtype = torch.bfloat16

    def load_quantized(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> None:
        expected = (self.num_embeddings, self.embedding_dim)
        if tuple(weight.shape) != expected or weight.dtype != torch.int8:
            raise ValueError(f"INT8 embedding must be int8 {expected}, got {weight.dtype} {tuple(weight.shape)}")
        self.weight = nn.Parameter(weight.to(device=device), requires_grad=False)
        self.weight_scale = weight_scale.to(device=device, dtype=torch.float32)
        self.compute_dtype = compute_dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        selected = F.embedding(input_ids, self.weight).to(self.compute_dtype)
        scale = self.weight_scale
        if scale.numel() == 1:
            selected_scale = scale.reshape(()).to(dtype=self.compute_dtype)
        elif scale.shape[0] == self.num_embeddings and scale.numel() == self.num_embeddings:
            # ComfyUI may store tensorwise-INT8 embedding scales per vocabulary row as
            # [num_embeddings, 1]. Select only the requested rows, just like the packed weight.
            selected_scale = F.embedding(input_ids, scale.reshape(self.num_embeddings, 1)).to(
                dtype=self.compute_dtype
            )
        else:
            raise ValueError(
                "INT8 embedding scale must be scalar or contain one value per vocabulary row, "
                f"got {tuple(scale.shape)}"
            )
        return selected * selected_scale
