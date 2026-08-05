from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from backends import krea2_face, minimax_h3_face


def validate_face_recipe_for_mode(mode: str, face_config: dict[str, Any]) -> None:
    try:
        resolution = int(face_config.get("resolution", 512))
        blocks = int(face_config.get("blocks_to_swap", 35 if mode == "MiniMax H3 (Experimental)" else 10))
        denoise_steps = int(face_config.get("denoise_steps", 12))
        draft_k = int(face_config.get("draft_k", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Face Refinement resolution, block swap, and step values must be whole numbers.") from exc
    if denoise_steps < 1 or not 1 <= draft_k <= denoise_steps:
        raise ValueError("Face Refinement DRaFT K must be between 1 and the denoising step count.")
    if mode == "MiniMax H3 (Experimental)":
        if resolution < 32 or resolution % 32:
            raise ValueError("MiniMax H3 Face Refinement resolution must be a positive multiple of 32.")
        if not 30 <= blocks <= 48:
            raise ValueError("MiniMax H3 Face Refinement needs 30–48 swapped blocks on a 24 GB GPU; start at 35.")
    elif resolution < 16 or resolution % 16 or not 0 <= blocks <= 26:
        raise ValueError("Krea 2 Face Refinement resolution must be a multiple of 16 and blocks to swap must be 0–26.")


def validate_face_environment(face_config: dict[str, Any]) -> None:
    missing = [name for name in ("onnx", "onnxruntime", "insightface") if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(
            "Face Refinement dependencies are missing: "
            + ", ".join(missing)
            + '. Install them with: pip install -e ".[face_refinement]"'
        )
    from musubi_tuner.face_refinement.face_models import models_complete

    model_dir = str(face_config.get("face_model_dir", "")).strip()
    if not models_complete(model_dir):
        raise ValueError(f"Face Refinement model files are incomplete: {model_dir or 'no folder configured'}")


def prepare_face_stage(
    base_settings: dict[str, Any],
    stage: dict[str, Any],
    index: int,
    input_lora: Path,
) -> tuple[dict[str, Any], list[str], Path]:
    mode = base_settings.get("training_mode")
    if mode not in {"Krea 2", "MiniMax H3 (Experimental)"}:
        raise ValueError("Face Refinement stages require Krea 2 or MiniMax H3 mode.")
    face_backend = minimax_h3_face if mode == "MiniMax H3 (Experimental)" else krea2_face
    face_config = copy.deepcopy(base_settings.get("face_refinement_config") or {})
    validate_face_recipe_for_mode(str(mode), face_config)
    if not face_config.get("preflight_report"):
        raise ValueError("Face Refinement needs a completed face-analysis preflight report.")
    if not input_lora.is_file():
        raise FileNotFoundError(f"Face Refinement input LoRA does not exist: {input_lora}")

    stage_steps = str(stage.get("steps", "")).strip()
    if not stage_steps.isdigit() or int(stage_steps) < 1:
        raise ValueError("Face Refinement stages need a positive step count.")
    face_config["steps"] = int(stage_steps)
    label = _stage_label(stage, index)
    output_path = face_backend.output_path(base_settings, label)
    if output_path.resolve() == input_lora.resolve():
        raise ValueError("Face Refinement output would overwrite its input LoRA.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_payload = _prompt_payload(face_config)
    prompts_path = output_path.parent / f"face_refinement_prompts_{index + 1}.json"
    prompts_path.write_text(json.dumps(prompt_payload, indent=2), encoding="utf-8")
    manifest_path = output_path.parent / f"face_refinement_references_{index + 1}.json"
    manifest_path.write_text(json.dumps({"reference_images": _reference_entries(face_config)}, indent=2), encoding="utf-8")
    face_config["reference_manifest"] = str(manifest_path)

    settings = dict(base_settings)
    settings["python_executable"] = sys.executable or "python"
    settings["stage_type"] = "face_refinement"
    settings["output_name"] = f"{base_settings['output_name']}-{label}"
    settings["max_train_steps"] = stage_steps
    settings["max_train_epochs"] = ""
    settings["face_output_path"] = str(output_path)
    settings["resume_path"] = ""
    settings["network_weights"] = str(input_lora)
    settings["resume_exact_position"] = False
    settings["recovery_mode"] = False
    settings["dop_enabled"] = False
    command = face_backend.build_command(settings, face_config, input_lora, output_path, prompts_path)
    return settings, command, output_path


def _stage_label(stage: dict[str, Any], index: int) -> str:
    from modern_gui.stages import stage_label

    return stage_label(stage, index)


def _prompt_payload(face_config: dict[str, Any]) -> dict[str, Any]:
    from musubi_tuner.face_refinement.lora_validation import render_trigger_prompts

    trigger = face_config.get("trigger_word", "")
    pose_plan = copy.deepcopy(face_config.get("pose_plan") or {})
    if face_config.get("pose_aware") and pose_plan.get("enabled"):
        from musubi_tuner.face_refinement.pose_plan import normalize_pose_plan, weighted_prompt_records

        excluded = set(face_config.get("excluded_reference_images") or [])
        counts: dict[str, int] = {}
        for item in face_config.get("preflight_report", {}).get("scored_images", []):
            if item.get("path") not in excluded:
                bucket = item.get("bucket", "uncertain")
                counts[bucket] = counts.get(bucket, 0) + 1
        pose_plan, warnings = normalize_pose_plan(
            pose_plan,
            counts,
            int(face_config.get("pose_min_references", 2)),
        )
        records = weighted_prompt_records(pose_plan)
        rendered = render_trigger_prompts([item["prompt"] for item in records], trigger)
        for item, prompt in zip(records, rendered):
            item["prompt"] = prompt
        return {"prompts": rendered, "prompt_records": records, "pose_plan": pose_plan, "warnings": warnings}
    prompts = face_config.get("prompts") or []
    if not prompts:
        raise ValueError("Face Refinement needs at least one prompt.")
    return {"prompts": render_trigger_prompts(prompts, trigger)}


def _reference_entries(face_config: dict[str, Any]) -> list[dict[str, Any]]:
    excluded = set(face_config.get("excluded_reference_images") or [])
    entries = [
        {
            "path": item["path"],
            "pose": item.get("bucket", "uncertain"),
            "pose_confidence": item.get("confidence", 0.0),
            "enabled": item["path"] not in excluded,
        }
        for item in face_config.get("preflight_report", {}).get("scored_images", [])
        if item.get("path")
    ]
    if not any(item["enabled"] for item in entries):
        raise ValueError("No enabled face references remain after exclusions.")
    return entries
