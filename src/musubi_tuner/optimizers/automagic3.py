"""Musubi-compatible integration of Ostris's Automagic3 (see vendored license).

Keep ordinary backward/clip/step ordering. The upstream fused hooks bypass
Accelerate's accumulation, clipping and overflow checks, so they are not used.
"""
import torch

from ._automagic3 import Automagic3 as _UpstreamAutomagic3
from .config import validate_automagic3


class Automagic3(_UpstreamAutomagic3):
    def __init__(self, params, lr=1e-4, fused=False, **kwargs):
        validate_automagic3(lr, fused=fused, **kwargs)
        groups = list(params)
        for entry in groups:
            parameters = entry["params"] if isinstance(entry, dict) else [entry]
            if isinstance(entry, dict):
                parameters = entry["params"] = list(parameters)
            if any(p.dtype != torch.float32 for p in parameters):
                raise ValueError("Automagic3 needs FP32 trainable adapter weights. Disable full_bf16/full_fp16; mixed precision BF16 is supported.")
        super().__init__(groups, lr=lr, fused=False, **kwargs)

    def load_state_dict(self, state_dict):
        if any(not {"beta2", "min_lr", "max_lr", "polarity_history"} <= group.keys()
               for group in state_dict.get("param_groups", [])):
            raise ValueError("This full state contains a different optimizer. Continue from LoRA weights to switch to Automagic3.")
        return super().load_state_dict(state_dict)


class Automagic3Scheduler(torch.optim.lr_scheduler.LRScheduler):
    """Report the optimizer's adapted LR without applying an external schedule.

Automagic3 stores the effective LR inside per-parameter state; an ordinary
constant LambdaLR would report the starting LR instead of that effective LR.
This scheduler is checkpointable and can be wrapped normally by Accelerate.
"""
    def get_lr(self):
        return self.optimizer.get_learning_rates()
