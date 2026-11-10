"""Non-fused, FP32 Musubi integration of Ostris Automagic4."""
import math
import torch

from ._automagic4 import Automagic4 as UpstreamAutomagic4
from .config import validate_automagic3


class Automagic4(UpstreamAutomagic4):
    def __init__(self, params, lr=1e-6, fused=False, sensitivity_quantile=0.9, step_target=0.1, **kwargs):
        validate_automagic3(lr, fused=fused, **kwargs)
        if not math.isfinite(sensitivity_quantile) or not 0 < sensitivity_quantile <= 1:
            raise ValueError("Automagic4 sensitivity_quantile must be in (0, 1].")
        if not math.isfinite(step_target) or not 0 < step_target < 2:
            raise ValueError("Automagic4 step_target must be in (0, 2).")
        groups = list(params)
        for entry in groups:
            parameters = entry["params"] if isinstance(entry, dict) else [entry]
            if isinstance(entry, dict):
                parameters = entry["params"] = list(parameters)
            if any(p.dtype != torch.float32 for p in parameters):
                raise ValueError("Automagic4 needs FP32 adapter weights; mixed precision BF16 is supported.")
        super().__init__(groups, lr=lr, fused=False, sensitivity_quantile=sensitivity_quantile,
                         step_target=step_target, **kwargs)

    def state_dict(self):
        state = super().state_dict()
        state["musubi_automagic4"] = {"sensitivity_quantile": self.sensitivity_quantile,
                                     "step_target": self.step_target}
        return state

    def load_state_dict(self, state_dict):
        options = state_dict.get("musubi_automagic4")
        if options != {"sensitivity_quantile": self.sensitivity_quantile, "step_target": self.step_target}:
            raise ValueError("Keep Automagic4 and its sensitivity settings when resuming a full state.")
        return super().load_state_dict(state_dict)
