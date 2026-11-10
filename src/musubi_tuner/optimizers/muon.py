"""Validated Fizgig-style batched Muon with a fallback for non-matrix parameters."""
import math
import torch

from ._muon import BatchedMuon, Muon as TensorMuon

DEFAULT_ARGS = dict(adjust_lr_fn="match_rms_adamw", weight_decay=0.01,
                    momentum=0.95, nesterov=True, ns_steps=5)


def validate_muon(lr, weight_decay=0.01, momentum=0.95, nesterov=True,
                  ns_coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7,
                  ns_steps=5, adjust_lr_fn="match_rms_adamw"):
    for name, value in (("lr", lr), ("weight_decay", weight_decay), ("momentum", momentum)):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"Muon {name} must be finite and non-negative.")
    if not isinstance(nesterov, bool):
        raise ValueError("Muon nesterov must be True or False.")
    if not isinstance(ns_steps, int) or isinstance(ns_steps, bool) or not 1 <= ns_steps < 100:
        raise ValueError("Muon ns_steps must be a whole number from 1 to 99.")
    if len(ns_coefficients) != 3 or not all(math.isfinite(float(v)) for v in ns_coefficients):
        raise ValueError("Muon ns_coefficients needs three finite numbers.")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("Muon eps must be finite and positive.")
    if adjust_lr_fn not in (None, "original", "match_rms_adamw", "spectral_unclamped"):
        raise ValueError("Choose a supported Muon adjust_lr_fn; Fizgig uses match_rms_adamw.")


class Muon(BatchedMuon, TensorMuon):
    """Batch the matrices; REPA biases / higher-dimensional adapters retain Muon's fallback.

    The original parameter groups remain intact for Accelerate, schedulers and
    saved states. Only a step's views of them are partitioned by dimensionality.
    """
    def __init__(self, params, lr=1e-4, **kwargs):
        options = {**DEFAULT_ARGS, **kwargs}
        validate_muon(lr, **options)
        # TensorMuon's constructor accepts the mixed parameter shapes, whereas
        # BatchedMuon's constructor deliberately checks for all-2D input.
        TensorMuon.__init__(self, params, lr=lr, **options)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = self.param_groups
        for group in groups:
            if any(p.grad is not None and p.grad.is_sparse for p in group["params"]):
                raise RuntimeError("Muon does not support sparse gradients.")
        try:
            self.param_groups = [{**g, "params": [p for p in g["params"] if p.ndim == 2]} for g in groups]
            BatchedMuon.step(self)
            self.param_groups = [{**g, "params": [p for p in g["params"] if p.ndim != 2]} for g in groups]
            TensorMuon.step(self)
        finally:
            self.param_groups = groups
        return loss

    def load_state_dict(self, state_dict):
        if any("ns_steps" not in g or "momentum" not in g for g in state_dict.get("param_groups", [])):
            raise ValueError("This is not a Muon state. Continue from LoRA weights to change optimizers.")
        return super().load_state_dict(state_dict)
