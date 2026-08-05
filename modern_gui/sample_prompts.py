from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any


_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
MINIMAX_H3_FIVE_FRAME_PREVIEW = "Five-frame video (experimental)"


def minimax_h3_scheduled_preview_frames(settings: dict[str, Any]) -> int:
    value = str(settings.get("minimax_h3_training_preview_mode") or "").strip().lower()
    return 5 if value in {"five_frame", MINIMAX_H3_FIVE_FRAME_PREVIEW.lower()} else 1


def enabled_sample_prompts(settings: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = settings.get("sample_prompts_data")
    if not isinstance(prompts, list):
        return []
    return [
        dict(prompt)
        for prompt in prompts
        if isinstance(prompt, dict) and prompt.get("enabled", True)
    ]


def serialize_sample_prompt(prompt: dict[str, Any], mode: str) -> str:
    """Serialize one visual prompt using the classic GUI's Musubi flags."""

    line = " ".join(str(prompt.get("prompt") or "").split())
    is_krea2 = mode == "Krea 2"

    def add(flag: str, key: str) -> None:
        nonlocal line
        value = str(prompt.get(key, "")).strip()
        if value:
            line += f" --{flag} {value}"

    add("w", "width")
    add("h", "height")
    add("s", "steps")
    add("l" if is_krea2 else "g", "guidance")
    if is_krea2:
        add("mu", "mu")
        add("y1", "y1")
        add("y2", "y2")
    else:
        add("f", "frames")
        add("fs", "flow_shift")
        add("l", "cfg_scale")
    add("d", "seed")
    add("n", "neg")
    if not is_krea2:
        add("i", "image_path")
    return line


def planned_sample_prompt_path(settings: dict[str, Any]) -> Path:
    dataset = Path(str(settings.get("dataset_config") or "")).expanduser()
    output = Path(str(settings.get("output_dir") or "")).expanduser()
    if dataset.is_file():
        parent = dataset.parent
    elif output.is_dir():
        parent = output
    else:
        parent = Path(tempfile.gettempdir())

    raw_name = str(settings.get("output_name") or "training").strip() or "training"
    safe_name = _UNSAFE_FILENAME.sub("-", raw_name).strip(" .-") or "training"
    return parent / f"{safe_name}_sample_prompts.txt"


def prepare_sample_prompt_settings(
    settings: dict[str, Any],
    *,
    write: bool,
) -> dict[str, Any]:
    """Attach the prompt-file path expected by the existing backend adapters.

    ``sample_prompts_data`` is authoritative when present. Command previews get
    the same deterministic path without touching disk; real launches atomically
    write the enabled prompts before command construction.
    """

    prepared = dict(settings)
    if "sample_prompts_data" not in prepared:
        return prepared

    mode = str(prepared.get("training_mode") or "Wan 2.2")
    prompts = enabled_sample_prompts(prepared)
    if mode == "MiniMax H3 (Experimental)":
        # Keep the prompt card's native-video frame setting for its standalone
        # Preview button, while scheduled samples use the explicit run setting.
        scheduled_frames = minimax_h3_scheduled_preview_frames(prepared)
        prompts = [dict(prompt, frames=scheduled_frames) for prompt in prompts]
    lines = [
        serialize_sample_prompt(prompt, mode)
        for prompt in prompts
    ]
    lines = [line for line in lines if line.strip()]
    if not lines:
        prepared["sample_prompts"] = ""
        return prepared

    destination = planned_sample_prompt_path(prepared)
    prepared["sample_prompts"] = str(destination)
    if not write:
        return prepared

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return prepared


def validate_sample_prompts(settings: dict[str, Any]) -> list[str]:
    prompts = settings.get("sample_prompts_data")
    if not isinstance(prompts, list):
        return []

    errors: list[str] = []
    positive_integer_keys = {
        "width": "width",
        "height": "height",
        "steps": "steps",
        "frames": "frame count",
    }
    numeric_keys = {
        "guidance": "guidance",
        "cfg_scale": "CFG scale",
        "flow_shift": "flow shift",
        "mu": "Mu",
        "y1": "Y1",
        "y2": "Y2",
    }
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict) or not prompt.get("enabled", True):
            continue
        label = f"Sample prompt {index + 1}"
        if not str(prompt.get("prompt") or "").strip():
            errors.append(f"{label} is included but has no positive prompt.")
        for key, field_label in positive_integer_keys.items():
            value = str(prompt.get(key, "")).strip()
            if value and (not value.isdigit() or int(value) < 1):
                errors.append(f"{label} {field_label} must be a positive whole number.")
        seed = str(prompt.get("seed", "")).strip()
        if seed:
            try:
                int(seed)
            except ValueError:
                errors.append(f"{label} seed must be a whole number or blank.")
        for key, field_label in numeric_keys.items():
            value = str(prompt.get(key, "")).strip()
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                errors.append(f"{label} {field_label} must be numeric or blank.")
    return errors
