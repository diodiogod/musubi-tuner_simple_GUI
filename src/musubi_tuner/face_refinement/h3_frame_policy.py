"""CUDA-independent validation for opt-in H3 temporal face rewards."""

import math


def validate_frame_policy(mode, worst_frame_weight):
    if mode not in {"one_frame", "five_frame"}:
        raise ValueError("MiniMax H3 training frames must be one_frame or five_frame")
    try:
        weight = float(worst_frame_weight)
    except (TypeError, ValueError):
        raise ValueError("Worst-frame reward weight must be a finite number between 0 and 1") from None
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Worst-frame reward weight must be a finite number between 0 and 1")
