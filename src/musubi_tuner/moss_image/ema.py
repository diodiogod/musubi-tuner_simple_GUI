"""Optional adapter-only EMA: averaged exports, live weights in resume states."""
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

STATE_KIND = "moss-adapter-ema-v1"
MANIFEST = "moss_ema.json"


def validate_ema_args(args):
    if args.moss_ema_enabled and (not math.isfinite(args.moss_ema_decay) or not 0 <= args.moss_ema_decay < 1):
        raise ValueError("MOSS EMA decay must be finite, at least 0 and less than 1. Try 0.98.")


def validate_ema_resume(resume_dir, enabled, decay):
    path = Path(resume_dir) / MANIFEST
    if path.is_file() != bool(enabled):
        raise ValueError("Full-state resume must keep EMA enabled/disabled as in the saved run. To change it, continue from LoRA weights with a fresh optimizer.")
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("kind") != STATE_KIND or manifest.get("decay") != decay:
            raise ValueError("The saved MOSS EMA configuration differs. Restore its decay or continue from LoRA weights.")


class AdapterEMA:
    """Fixed-decay FP32 EMA over trainable adapter parameters only.

    Matches the ordinary EMA used by AI Toolkit (use_num_updates=False):
    shadow = decay * shadow + (1 - decay) * live. This is weight smoothing,
    not an additional loss or a guarantee against overtraining.
    """
    def __init__(self, network, decay=0.98, device="training"):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite, at least 0 and less than 1.")
        self.parameters = {name: p for name, p in network.named_parameters() if p.requires_grad}
        if not self.parameters:
            raise ValueError("EMA needs trainable adapter parameters.")
        self.decay, self.num_updates = float(decay), 0
        self.shadow = {name: p.detach().to(device=p.device if device == "training" else "cpu",
                                          dtype=torch.float32, copy=True)
                       for name, p in self.parameters.items()}
        self._swapped = False

    @torch.no_grad()
    def update(self):
        if self._swapped:
            raise RuntimeError("Cannot update EMA while averaged weights are active.")
        for name, p in self.parameters.items():
            shadow = self.shadow[name]
            shadow.lerp_(p.detach().to(device=shadow.device, dtype=shadow.dtype), 1 - self.decay)
        self.num_updates += 1

    @contextmanager
    def average_parameters(self):
        if self._swapped:
            yield
            return
        # Backups live in system RAM, avoiding another adapter-sized VRAM copy.
        live = {name: p.detach().to("cpu", copy=True) for name, p in self.parameters.items()}
        self._swapped = True
        try:
            with torch.no_grad():
                for name, p in self.parameters.items():
                    p.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, p in self.parameters.items():
                    p.copy_(live[name])
            self._swapped = False

    def save_manifest(self, models, weights, output_dir):
        Path(output_dir, MANIFEST).write_text(
            json.dumps({"kind": STATE_KIND, "decay": self.decay}), encoding="utf-8",
        )

    def state_dict(self):
        return {"kind": STATE_KIND, "decay": self.decay, "num_updates": self.num_updates,
                "shadow": self.shadow}

    def load_state_dict(self, state):
        if state.get("kind") != STATE_KIND or state.get("decay") != self.decay:
            raise ValueError("Saved EMA configuration does not match this run.")
        saved = state.get("shadow", {})
        if saved.keys() != self.shadow.keys():
            raise ValueError("Saved EMA adapter layers differ from this network.")
        for name, value in saved.items():
            if value.shape != self.shadow[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid saved EMA tensor: {name}")
        updates = state.get("num_updates")
        if not isinstance(updates, int) or updates < 0:
            raise ValueError("Invalid EMA update count.")
        for name, value in saved.items():
            self.shadow[name].copy_(value)
        self.num_updates = updates
