"""Optional low-precision LoRA compute, with FP32 master weights and gradients.

Adapted from Fizgig fe1fb5d families/lora.py (_LoraLowp), Apache-2.0.
Copyright the Fizgig contributors; see docs/THIRD_PARTY_PERFORMANCE.md.
"""
import torch
from torch.nn import functional as F
from musubi_tuner.networks.lora import LoRAModule


class LowPrecisionLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, down, up, scale):
        hidden = F.linear(x, down.to(x.dtype))
        output = F.linear(hidden, up.to(x.dtype)) * scale
        ctx.save_for_backward(x, hidden, down, up)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, gradient):
        x, hidden, down, up = ctx.saved_tensors
        gradient = gradient.to(x.dtype) * ctx.scale
        middle = gradient @ up.to(x.dtype)
        g, h, m, inputs = (v.reshape(-1, v.shape[-1]) for v in (gradient, hidden, middle, x))
        return (middle @ down.to(x.dtype), (m.T @ inputs).to(down.dtype),
                (g.T @ h).to(up.dtype), None)


class FastMossLoRA(LoRAModule):
    def forward(self, x, base_output=None):
        # Dropout and split/convolutional adapters retain Musubi's existing path.
        # The common plain linear LoRA keeps only BF16 inputs/intermediates.
        if (not self.enabled or self.split_dims is not None or x.dtype not in (torch.bfloat16, torch.float16)
                or self.lora_down.weight.ndim != 2
                or any(v for v in (self.dropout, self.rank_dropout, self.module_dropout))):
            return super().forward(x, base_output=base_output)
        base = self.org_forward(x) if base_output is None else base_output
        delta = LowPrecisionLoRA.apply(x, self.lora_down.weight, self.lora_up.weight,
                                      self.multiplier * self.scale)
        return base + delta.to(base.dtype)
