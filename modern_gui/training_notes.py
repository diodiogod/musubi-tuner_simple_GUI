from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from modern_gui.stages import stage_label


def _enabled_number(settings: dict[str, Any], key: str) -> str | None:
    try:
        value = float(settings.get(key) or 0)
    except (TypeError, ValueError):
        return None
    return f"{value:g}" if value > 0 else None


def _stage_limit(stage: dict[str, Any]) -> str:
    steps = str(stage.get("steps") or "").strip()
    if steps:
        return f"{steps} steps"
    return f"{str(stage.get('epochs') or '?').strip()} epochs"


def _dataset_note_label(value: Any) -> str:
    source = Path(str(value or "")).expanduser()
    if not source.name:
        return ""
    label = source.stem
    try:
        payload = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return label

    resolutions: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "resolution":
                    text = (
                        "×".join(str(part) for part in child)
                        if isinstance(child, (list, tuple))
                        else str(child)
                    )
                    if text and text not in resolutions:
                        resolutions.append(text)
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(payload)
    return f"{label} ({'/'.join(resolutions)})" if resolutions else label


def training_settings_summary(settings: dict[str, Any]) -> str:
    parts = [str(settings.get("training_mode") or "Training")]
    output_name = str(settings.get("output_name") or "").strip()
    if output_name:
        parts.append(f"run={output_name}")

    network = str(settings.get("network_type") or "LoRA")
    rank = str(settings.get("network_dim_low") or "").strip()
    alpha = str(settings.get("network_alpha_low") or "").strip()
    network_text = network
    if rank:
        network_text += f" rank {rank}"
    if alpha:
        network_text += f" α{alpha}"
    parts.append(network_text)

    stages = [
        item
        for item in settings.get("staged_training_config", [])
        if isinstance(item, dict) and item.get("enabled", True)
    ]
    if settings.get("use_staged_training") and stages and not settings.get("stage_type"):
        stage_notes = [
            f"{stage_label(stage, index)} {_stage_limit(stage)}"
            + (" face" if stage.get("type", "standard") == "face_refinement" else "")
            for index, stage in enumerate(stages)
        ]
        parts.append("staged " + " → ".join(stage_notes))
    else:
        steps = str(settings.get("max_train_steps") or "").strip()
        epochs = str(settings.get("max_train_epochs") or "").strip()
        if steps:
            parts.append(f"{steps} steps")
        elif epochs:
            parts.append(f"{epochs} epochs")
        dataset = _dataset_note_label(settings.get("dataset_config"))
        if dataset:
            parts.append(f"data={dataset}")

    learning_rate = str(settings.get("learning_rate") or "").strip()
    if learning_rate:
        parts.append(f"lr={learning_rate}")
    optimizer = str(settings.get("optimizer_type") or "").strip()
    if optimizer:
        parts.append(f"opt={optimizer}")

    if settings.get("dop_enabled"):
        strength = _enabled_number(settings, "dop_loss_weight")
        class_word = str(settings.get("dop_class_word") or "").strip()
        text = f"DOP {strength or '?'}"
        if class_word:
            text += f" ({class_word})"
        parts.append(text)

    if str(settings.get("training_mode")) == "Krea 2":
        depth = _enabled_number(settings, "krea2_depth_anchor_weight")
        if depth:
            text = f"depth {depth}@{settings.get('krea2_depth_anchor_input_size') or 518}"
            text += " GPU" if settings.get("krea2_keep_depth_helpers_on_gpu") else " offload"
            parts.append(text)
        noise = _enabled_number(settings, "krea2_weight_noise_sigma")
        if noise:
            parts.append(
                f"weight-noise {noise} {settings.get('krea2_weight_noise_mode') or 'relative'}"
            )
        projector = str(settings.get("krea2_projector_diff") or "").strip()
        if projector:
            strength = str(settings.get("krea2_projector_diff_strength") or "1").strip()
            parts.append(f"projector={Path(projector).name}@{strength}")

    blocks = str(settings.get("blocks_to_swap") or "").strip()
    if blocks and blocks != "0":
        parts.append(f"swap={blocks}")
    return "Settings: " + "; ".join(parts)


def effective_training_comment(settings: dict[str, Any]) -> str:
    custom = str(settings.get("training_comment") or "").strip()
    if not settings.get("auto_training_settings_summary"):
        return custom
    generated = training_settings_summary(settings)
    return "\n\n".join(part for part in (custom, generated) if part)
