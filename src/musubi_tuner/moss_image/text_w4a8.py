"""ComfyUI grouped W4A8 text weights, decoded into temporary ConvRot INT8.

The on-disk nibble order and grouped scale calculation follow Comfy Kitchen's
Apache-2.0 eager W4A8 implementation. This fallback keeps the packed weights
in RAM/VRAM and decodes one projection at a time; it does not depend on ComfyUI.
"""
import torch
from torch import nn
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8LinearFn


def decode_rotated_int8(packed, group_scale, codebook, group_size, *, row_chunk=512):
    n, packed_k = packed.shape
    k = packed_k * 2
    if group_scale.shape != (n, k // group_size) or k % group_size:
        raise ValueError("Invalid W4A8 group scale dimensions.")
    output = torch.empty((n, k), dtype=torch.int8, device=packed.device)
    for first in range(0, n, row_chunk):
        last = min(first + row_chunk, n)
        data = packed[first:last].view(torch.uint8).to(torch.int32)
        codes = torch.stack((data & 15, (data >> 4) & 15), dim=-1).reshape(last - first, k).long()
        values = codebook[codes] if codebook is not None else codes.float() - 8
        values = values.reshape(last - first, k // group_size, group_size)
        values = values * group_scale[first:last].float().unsqueeze(-1)
        output[first:last] = values.round().clamp_(-127, 127).to(torch.int8).reshape(last - first, k)
    return output


def patch_w4a8_linear(module, state, path, spec):
    if not isinstance(module, nn.Linear):
        raise ValueError(f"W4A8 tensor {path} does not name a Linear layer")
    n, k = module.out_features, module.in_features
    group = int(spec.get("group_size", 16))
    rotation = int(spec.get("convrot_groupsize", 256))
    if not spec.get("convrot") or k % group or k % rotation:
        raise ValueError(f"Unsupported W4A8 settings for {path}")
    weight = state[path + ".weight"]
    scale = state[path + ".w4_scale"]
    channel = state[path + ".w4_channel"]
    if weight.dtype != torch.int8 or tuple(weight.shape) != (n, k // 2):
        raise ValueError(f"Invalid W4A8 packed weights for {path}")
    if tuple(scale.shape) != (n, k // group) or tuple(channel.shape) != (n,):
        raise ValueError(f"Invalid W4A8 scales for {path}")
    module.weight = nn.Parameter(torch.empty_like(weight, device="meta"), requires_grad=False)
    module.register_buffer("w4_scale", torch.empty_like(scale, device="meta"))
    module.register_buffer("w4_channel", torch.empty_like(channel, device="meta"))
    if path + ".w4_codebook" in state:
        if tuple(state[path + ".w4_codebook"].shape) != (16,):
            raise ValueError(f"Invalid W4A8 codebook for {path}")
        module.register_buffer("w4_codebook", torch.empty_like(state[path + ".w4_codebook"], device="meta"))
    else:
        module.register_buffer("w4_codebook", None)

    def forward(self, x):
        quantized = decode_rotated_int8(self.weight, self.w4_scale, self.w4_codebook, group)
        return ConvRotInt8LinearFn.apply(x, quantized, self.w4_channel, self.bias, rotation, "bf16")

    module.forward = forward.__get__(module, type(module))
