"""Lightweight validation shared by GUI preflight and optimizer construction."""
import math
import json
from pathlib import Path

OPTIMIZER_MANIFEST = "moss_optimizer.json"


def validate_optimizer_resume(folder, optimizer_type):
    path = Path(folder) / OPTIMIZER_MANIFEST
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("optimizer_type") != str(optimizer_type).lower():
            raise ValueError("Full-state resume must keep the same optimizer. To change optimizers, continue from LoRA weights with a fresh optimizer.")


def validate_automagic3(lr, min_lr=1e-8, max_lr=1e3, beta2=0.999, eps=1e-30,
                       clip_threshold=1.0, weight_decay=0.0, polarity_history=8, fused=False):
    if fused:
        raise ValueError("Automagic3 uses fused=False in Musubi to preserve gradient accumulation and clipping.")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in (lr, min_lr, max_lr)):
        raise ValueError("Automagic3 learning rate and bounds must be finite and positive.")
    if not min_lr <= lr <= max_lr:
        raise ValueError("Automagic3 starting learning rate must be between min_lr and max_lr.")
    if not 0 <= beta2 < 1:
        raise ValueError("Automagic3 beta2 must be between 0 (inclusive) and 1 (exclusive).")
    for key, value in (("eps", eps), ("clip_threshold", clip_threshold)):
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"Automagic3 {key} must be finite and positive.")
    if not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise ValueError("Automagic3 weight_decay must be finite and non-negative.")
    if not isinstance(polarity_history, int) or not 2 <= polarity_history <= 64:
        raise ValueError("Automagic3 polarity_history must be a whole number from 2 to 64.")
