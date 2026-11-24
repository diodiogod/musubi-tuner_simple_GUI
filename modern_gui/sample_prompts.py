from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from backends.minimax_h3 import is_multimodal
from backends.minimax_h3_picture import MODE as H3_PICTURE_MODE


_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
MINIMAX_H3_FIVE_FRAME_PREVIEW = "Five-frame video (recommended)"


def h3_conditioning(settings, prompt, frames):
    task = settings.get("minimax_h3_multimodal_task") or "t2va"
    keys = {
        "fl2va": ("first_frame", "last_frame"),
        "ref2va": ("reference_jsonl",),
    }.get(task, ())
    if keys and frames < 5:
        raise ValueError("Choose a video length of at least five frames for conditioned MiniMax H3 previews.")
    values = {}
    for key in keys:
        raw = prompt.get(key) or (prompt.get("image_path") if key == "first_frame" else "")
        path = Path(str(raw or "")).expanduser()
        if not path.is_file() or (key == "reference_jsonl" and path.suffix.lower() != ".jsonl"):
            raise ValueError(f"Choose an existing {key.replace('_', ' ')} for the {task.upper()} sample.")
        values[key] = str(path.resolve())
    if task == "ref2va":
        index = str(prompt.get("reference_index") or "0")
        if not index.isdigit():
            raise ValueError("The reference manifest row must be a nonnegative whole number, starting at 0.")
        values["reference_index"] = int(index)
    return values


def _native_h3_prompt(settings, prompt, frames):
    from musubi_tuner.training.sampling_prompts import line_to_prompt_dict

    fields = {key: value for key, value in prompt.items() if key not in {"image_path", "flow_shift"}}
    sample = line_to_prompt_dict(serialize_sample_prompt(dict(fields, frames=frames), "MiniMax H3 (Experimental)"))
    sample.update(h3_conditioning(settings, prompt, frames))
    sample["allow_experimental_duration"] = frames == 5 or bool(settings.get("minimax_h3_allow_experimental_duration"))
    return sample


def minimax_h3_scheduled_preview_frames(settings: dict[str, Any]) -> int:
    value = str(settings.get("minimax_h3_training_preview_mode") or MINIMAX_H3_FIVE_FRAME_PREVIEW).strip().lower()
    return 1 if value in {"one_frame", "one frame (safe)", "one frame (low memory)"} else 5


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
    is_moss = mode == "MOSS-Image (Experimental)"

    def add(flag: str, key: str) -> None:
        nonlocal line
        value = str(prompt.get(key, "")).strip()
        if value:
            line += f" --{flag} {value}"

    add("w", "width")
    add("h", "height")
    add("s", "steps")
    add("l" if is_krea2 or is_moss else "g", "guidance")
    if is_krea2:
        add("mu", "mu")
        add("y1", "y1")
        add("y2", "y2")
    elif is_moss:
        add("fs", "flow_shift")
        add("moss_sampler", "sampler")
        add("moss_scheduler", "scheduler")
    else:
        add("f", "frames")
        add("fs", "flow_shift")
        add("l", "cfg_scale")
    add("d", "seed")
    add("n", "neg")
    if not is_krea2 and not is_moss:
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
    native_h3 = mode == "MiniMax H3 (Experimental)" and is_multimodal(prepared)
    if mode == "MiniMax H3 (Experimental)":
        # Keep the prompt card's native-video frame setting for its standalone
        # Preview button, while scheduled samples use the explicit run setting.
        scheduled_frames = minimax_h3_scheduled_preview_frames(prepared)
        prompts = [dict(prompt, frames=scheduled_frames) for prompt in prompts]
    elif mode == H3_PICTURE_MODE:
        prompts = [dict(prompt, frames=1, steps=6, guidance=1, cfg_scale=1, flow_shift="", image_path="")
                   for prompt in prompts]
    lines = [serialize_sample_prompt(prompt, mode) for prompt in prompts]
    lines = [line for line in lines if line.strip()]
    if not lines:
        prepared["sample_prompts"] = ""
        return prepared

    destination = planned_sample_prompt_path(prepared)
    if native_h3:
        destination = destination.with_suffix(".json")
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
            if native_h3:
                json.dump([_native_h3_prompt(prepared, prompt, scheduled_frames) for prompt in prompts], stream,
                          ensure_ascii=False, indent=2)
            else:
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
        if settings.get("training_mode") == "MiniMax H3 (Experimental)" and is_multimodal(settings):
            try:
                h3_conditioning(settings, prompt, minimax_h3_scheduled_preview_frames(settings))
            except ValueError as exc:
                errors.append(f"{label}. {exc}")
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
