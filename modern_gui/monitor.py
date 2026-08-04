from __future__ import annotations

import math
import re
from typing import Any


METRIC_PATTERN = re.compile(r"(?P<key>avr_loss|loss/current|loss/depth_anchor|loss/dop_weighted|loss/dop)=([-+\d.eE]+)")
STEP_PATTERN = re.compile(r"^\s*steps:.*?\b(?P<step>\d+)\s*/\s*(?P<total>\d+)\b", re.IGNORECASE)
EPOCH_PATTERN = re.compile(r"\bepoch\s*=?\s*(?P<epoch>\d+)\s*/\s*(?P<total>\d+)", re.IGNORECASE)
FACE_STEP_PATTERN = re.compile(r"^step=(?P<step>\d+)/(?P<total>\d+)")


def is_training_progress_line(line: str) -> bool:
    """Return whether a line is the replaceable main trainer progress bar."""

    return STEP_PATTERN.search(line) is not None


def parse_training_line(line: str) -> dict[str, Any]:
    update: dict[str, Any] = {}
    step_match = STEP_PATTERN.search(line) or FACE_STEP_PATTERN.search(line)
    if step_match:
        update["step"] = int(step_match.group("step"))
        update["total_steps"] = int(step_match.group("total"))
    epoch_match = EPOCH_PATTERN.search(line)
    if epoch_match:
        update["epoch"] = int(epoch_match.group("epoch"))
        update["total_epochs"] = int(epoch_match.group("total"))
    for match in METRIC_PATTERN.finditer(line):
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        key = match.group("key")
        if key in {"avr_loss", "loss/current"}:
            update["loss"] = value
        elif key == "loss/depth_anchor":
            update["depth_loss"] = value
        elif key == "loss/dop":
            update["dop_loss"] = value
        elif key == "loss/dop_weighted":
            update["dop_weighted"] = value
    return update


def gpu_snapshot() -> dict[str, Any]:
    try:
        import pynvml
    except ImportError:
        return {"available": False, "message": "pynvml is not installed", "devices": []}
    try:
        pynvml.nvmlInit()
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            devices.append(
                {
                    "index": index,
                    "name": str(name),
                    "memory_used": int(memory.used),
                    "memory_total": int(memory.total),
                    "memory_percent": round(memory.used / memory.total * 100, 1) if memory.total else 0,
                    "gpu_percent": int(utilization.gpu),
                }
            )
        return {"available": True, "devices": devices}
    except Exception as exc:
        return {"available": False, "message": str(exc), "devices": []}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
