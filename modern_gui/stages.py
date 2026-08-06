from __future__ import annotations

import re
from pathlib import Path
from typing import Any

STAGE_OPTIMIZER_OVERRIDE_KEYS = (
    "learning_rate", "optimizer_type", "lr_scheduler", "lr_warmup_steps", "timestep_sampling",
)


def stage_label(stage: dict[str, Any], index: int = 0) -> str:
    label = str(stage.get("label", "")).strip() or f"stage-{index + 1}"
    label = f"{label}px" if label.isdigit() else label
    # Artifact names must remain one path component on Windows.
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", label).strip(" .-") or f"stage-{index + 1}"


def enabled_stages(settings: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(stage)
        for stage in settings.get("staged_training_config", [])
        if stage.get("enabled", True)
    ]


def validate_stage_plan(settings: dict[str, Any]) -> list[dict[str, Any]]:
    stages = enabled_stages(settings)
    if not stages:
        raise ValueError("Staged progression is enabled but no enabled stages exist.")
    if settings.get("training_mode") == "Wan 2.2":
        separate_wan_runs = (
            settings.get("train_low_noise")
            and settings.get("train_high_noise")
            and (
                str(settings.get("network_dim_high") or "").strip()
                or str(settings.get("network_alpha_high") or "").strip()
            )
        )
        if separate_wan_runs:
            raise ValueError(
                "Staged continuation cannot map one state to separate Wan low/high-noise runs. "
                "Use a combined run or stage each noise model separately."
            )
    seen_labels: set[str] = set()
    invalid_label_chars = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    for index, stage in enumerate(stages):
        raw_label = str(stage.get("label") or "").strip()
        if not raw_label:
            raise ValueError(f"Stage {index + 1} needs a label.")
        if invalid_label_chars.search(raw_label) or raw_label.endswith((".", " ")):
            raise ValueError(
                f"{raw_label or f'Stage {index + 1}'} has a label that is not safe for Windows filenames."
            )
        normalized_label = stage_label(stage, index).casefold()
        if normalized_label in seen_labels:
            raise ValueError(f"Stage labels must be unique: {raw_label}")
        seen_labels.add(normalized_label)
        stage_type = stage.get("type", "standard")
        if stage_type not in {"standard", "face_refinement"}:
            raise ValueError(f"{stage_label(stage, index)} has an unsupported stage type: {stage_type}")
        limit = str(stage.get("steps", "")).strip() or str(stage.get("epochs", "")).strip()
        if not limit.isdigit() or int(limit) < 1:
            raise ValueError(f"{stage_label(stage, index)} needs a positive epoch or step limit.")
        if stage_type == "standard":
            handoff_mode = str(stage.get("handoff_mode", "state"))
            if handoff_mode not in {"state", "weights"}:
                raise ValueError(f"{stage_label(stage, index)} has an unsupported handoff mode: {handoff_mode}")
            has_standard_predecessor = index > 0 and stages[index - 1].get("type", "standard") == "standard"
            if has_standard_predecessor and handoff_mode == "state" and any(str(stage.get(key, "")).strip() for key in STAGE_OPTIMIZER_OVERRIDE_KEYS):
                raise ValueError(
                    f"{stage_label(stage, index)} has optimizer overrides but preserves the previous optimizer. "
                    "Choose LoRA weights + fresh optimizer for this stage."
                )
            learning_rate = str(stage.get("learning_rate", "")).strip()
            if learning_rate:
                try:
                    if float(learning_rate) <= 0:
                        raise ValueError
                except ValueError as exc:
                    raise ValueError(f"{stage_label(stage, index)} learning rate must be greater than zero.") from exc
            dataset = Path(str(stage.get("dataset_config", ""))).expanduser()
            if not dataset.is_file():
                raise ValueError(f"{stage_label(stage, index)} has no valid dataset TOML: {dataset}")
            dop_mode = str(stage.get("dop_mode", "inherit"))
            if dop_mode not in {"inherit", "enable", "disable"}:
                raise ValueError(f"{stage_label(stage, index)} has an unsupported DOP behavior: {dop_mode}")
            effective = dict(settings)
            apply_dop_overrides(effective, stage)
            if effective.get("dop_enabled"):
                if settings.get("training_mode") not in {"Krea 2", "Flux.2 Klein", "MiniMax H3 (Experimental)"}:
                    raise ValueError(
                        f"{stage_label(stage, index)} enables DOP, but DOP is supported only for "
                        "Krea 2, MiniMax H3, and FLUX.2 Klein."
                    )
                try:
                    from musubi_tuner.training.dop import validate_dop_config

                    strength = float(effective.get("dop_loss_weight") or 0)
                    validate_dop_config(
                        effective.get("dop_trigger_word", ""),
                        effective.get("dop_class_word", ""),
                        strength,
                    )
                    if strength <= 0:
                        raise ValueError("DOP preservation strength must be greater than zero.")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{stage_label(stage, index)} DOP configuration: {exc}") from exc
            depth_mode = str(stage.get("depth_helpers_mode", "inherit"))
            if depth_mode not in {"inherit", "keep on GPU", "offload to CPU"}:
                raise ValueError(
                    f"{stage_label(stage, index)} has an unsupported depth-helper policy: {depth_mode}"
                )
            if depth_mode != "inherit" and settings.get("training_mode") not in {"Krea 2", "MiniMax H3 (Experimental)"}:
                raise ValueError(
                    f"{stage_label(stage, index)} can override depth-helper memory only in Krea 2 or MiniMax H3."
                )
        elif settings.get("training_mode") not in {"Krea 2", "MiniMax H3 (Experimental)"}:
            raise ValueError("Face Refinement stages are supported only in Krea 2 or MiniMax H3 mode.")
        else:
            face_config = settings.get("face_refinement_config") or {}
            from modern_gui.face_stages import validate_face_recipe_for_mode

            validate_face_recipe_for_mode(str(settings.get("training_mode")), face_config)
            if index == 0:
                if face_config.get("input_mode") != "existing_lora":
                    raise ValueError(
                        "Face Refinement can be the first enabled stage only when its input mode is "
                        "“Refine an existing LoRA.”"
                    )
                input_lora = Path(str(face_config.get("input_lora") or "")).expanduser()
                if not input_lora.is_file():
                    raise ValueError(f"First-stage Face Refinement input LoRA does not exist: {input_lora}")
    return stages


def apply_dop_overrides(settings: dict[str, Any], stage: dict[str, Any]) -> None:
    mode = str(stage.get("dop_mode", "inherit"))
    if mode == "disable":
        settings["dop_enabled"] = False
        return
    if mode == "enable":
        settings["dop_enabled"] = True
    for key in ("dop_loss_weight", "dop_trigger_word", "dop_class_word"):
        value = str(stage.get(key, "")).strip()
        if value:
            settings[key] = value


def apply_depth_memory_override(settings: dict[str, Any], stage: dict[str, Any]) -> None:
    mode = str(stage.get("depth_helpers_mode", "inherit"))
    if mode == "keep on GPU":
        settings["krea2_keep_depth_helpers_on_gpu"] = True
    elif mode == "offload to CPU":
        settings["krea2_keep_depth_helpers_on_gpu"] = False


def apply_fresh_optimizer_overrides(settings: dict[str, Any], stage: dict[str, Any]) -> None:
    if str(stage.get("handoff_mode", "state")) != "weights":
        return
    for key in STAGE_OPTIMIZER_OVERRIDE_KEYS:
        value = str(stage.get(key, "")).strip()
        if value:
            settings[key] = value


def prepare_standard_stage(
    base_settings: dict[str, Any],
    stage: dict[str, Any],
    index: int,
    resume_path: str = "",
    network_weights: str | None = None,
) -> dict[str, Any]:
    settings = dict(base_settings)
    if network_weights is None:
        network_weights = str(base_settings.get("network_weights") or "")
    settings["dataset_config"] = str(stage["dataset_config"])
    stage_steps = str(stage.get("steps", "")).strip()
    if stage_steps:
        settings["max_train_steps"] = stage_steps
        settings["max_train_epochs"] = ""
    else:
        settings["max_train_steps"] = ""
        settings["max_train_epochs"] = str(stage["epochs"])
    settings["output_name"] = f"{base_settings['output_name']}-{stage_label(stage, index)}"
    settings["stage_type"] = "standard"
    apply_dop_overrides(settings, stage)
    apply_depth_memory_override(settings, stage)
    apply_fresh_optimizer_overrides(settings, stage)
    settings["save_state"] = True
    settings["recache_latents"] = bool(base_settings.get("staged_recache_latents", True))
    explicit_first_text_recache = index == 0 and bool(base_settings.get("recache_text"))
    force_text_recache = bool(base_settings.get("staged_recache_text")) or explicit_first_text_recache
    settings["recache_text"] = bool(settings.get("dop_enabled")) or force_text_recache
    settings["dop_cache_reuse"] = bool(settings.get("dop_enabled")) and not force_text_recache
    settings["resume_path"] = resume_path
    settings["resume_exact_position"] = False
    settings["recovery_mode"] = False
    settings["network_weights"] = "" if resume_path else network_weights
    return settings


def effective_run_name(settings: dict[str, Any]) -> str:
    run_name = str(settings["output_name"])
    if settings.get("training_mode") == "Wan 2.2":
        train_low = bool(settings.get("train_low_noise"))
        train_high = bool(settings.get("train_high_noise"))
        combined = train_low and train_high and not (
            str(settings.get("network_dim_high") or "").strip()
            or str(settings.get("network_alpha_high") or "").strip()
        )
        if not combined:
            run_name += "_HighNoise" if train_high else "_LowNoise"
    return run_name


def candidate_state_paths(settings: dict[str, Any]) -> list[Path]:
    run_name = effective_run_name(settings)
    base = Path(settings["output_dir"]) / run_name
    candidates: list[Path] = []
    epoch = str(settings.get("max_train_epochs", "")).strip()
    if epoch.isdigit():
        candidates.append(base / f"{run_name}-{int(epoch):06d}-state")
    candidates.append(base / f"{run_name}-state")
    return candidates


def candidate_lora_paths(settings: dict[str, Any]) -> list[Path]:
    run_name = effective_run_name(settings)
    base = Path(settings["output_dir"]) / run_name
    candidates: list[Path] = []
    epoch = str(settings.get("max_train_epochs", "")).strip()
    if epoch.isdigit():
        candidates.append(base / f"{run_name}-{int(epoch):06d}.safetensors")
    candidates.append(base / f"{run_name}.safetensors")
    return candidates


def resolve_standard_state(settings: dict[str, Any]) -> Path:
    state = next((path for path in candidate_state_paths(settings) if path.is_dir()), None)
    if state is None:
        expected = ", ".join(str(path) for path in candidate_state_paths(settings))
        raise FileNotFoundError(f"Expected staged state was not created. Checked: {expected}")
    return state


def resolve_stage_lora(settings: dict[str, Any]) -> Path:
    lora = next((path for path in candidate_lora_paths(settings) if path.is_file()), None)
    if lora is None:
        expected = ", ".join(str(path) for path in candidate_lora_paths(settings))
        raise FileNotFoundError(f"Expected staged LoRA was not created. Checked: {expected}")
    return lora
