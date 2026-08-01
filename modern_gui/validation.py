from __future__ import annotations

from pathlib import Path
from typing import Any

from modern_gui.recovery import validate_accelerate_state
from modern_gui.sample_prompts import enabled_sample_prompts, validate_sample_prompts


def validate_training_settings(settings: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def error(key: str, message: str) -> None:
        errors.append({"key": key, "message": message})

    def warning(key: str, message: str) -> None:
        warnings.append({"key": key, "message": message})

    def require(key: str, label: str) -> None:
        if not str(settings.get(key) or "").strip():
            error(key, f"{label} is required.")

    mode = settings.get("training_mode", "Wan 2.2")
    configured_stages = [
        stage
        for stage in settings.get("staged_training_config", [])
        if isinstance(stage, dict) and stage.get("enabled")
    ]
    first_stage = configured_stages[0] if configured_stages else {}
    face_config = settings.get("face_refinement_config") or {}
    refinement_only = bool(
        settings.get("use_staged_training")
        and first_stage.get("type") == "face_refinement"
        and face_config.get("input_mode") == "existing_lora"
    )
    if not settings.get("use_staged_training"):
        require("dataset_config", "Dataset configuration")
    dataset_path = Path(str(settings.get("dataset_config") or "")).expanduser()
    if (
        not settings.get("use_staged_training")
        and str(settings.get("dataset_config") or "").strip()
        and not dataset_path.is_file()
    ):
        error("dataset_config", f"Dataset TOML does not exist: {dataset_path}")
    require("output_dir", "Output directory")
    require("output_name", "Output name")
    require("vae_model", "VAE model")

    if mode == "Wan 2.2":
        if not settings.get("train_low_noise") and not settings.get("train_high_noise"):
            error("train_low_noise", "Enable at least one Wan noise model.")
        if settings.get("train_low_noise"):
            require("dit_low_noise", "Wan low-noise DiT")
        if settings.get("train_high_noise"):
            require("dit_high_noise", "Wan high-noise DiT")
        require("t5_model", "Wan T5 encoder")
        if settings.get("is_i2v"):
            require("clip_model", "Wan I2V CLIP model")
    elif mode in {"Flux.2 Klein", "Flux.2 Dev"}:
        require("flux2_dit_model", "FLUX.2 DiT")
        require("flux2_text_encoder", "FLUX.2 text encoder")
    elif mode == "Krea 2":
        require("krea2_dit_model", "Krea 2 RAW DiT")
        require("krea2_text_encoder", "Krea 2 text encoder")
        if settings.get("fp8_base") and not settings.get("fp8_scaled"):
            error("fp8_scaled", "Krea 2 FP8 Base requires FP8 Scaled.")
        if settings.get("krea2_turbo_dit_cache") and not settings.get("krea2_turbo_dit"):
            error("krea2_turbo_dit", "Turbo DiT caching requires a Turbo checkpoint.")
        if settings.get("krea2_turbo_dit") and str(settings.get("blocks_to_swap") or "").strip() not in {"", "0"}:
            error("blocks_to_swap", "Krea Turbo sampling cannot be combined with Blocks to Swap.")
    else:
        error("training_mode", f"Unsupported training mode: {mode}")

    for key, label, minimum in (
        ("learning_rate", "Learning rate", 0.0),
        ("gradient_accumulation_steps", "Gradient accumulation", 1.0),
        ("network_dim_low", "Network rank", 1.0),
    ):
        if refinement_only and key in {"learning_rate", "network_dim_low"}:
            continue
        value = str(settings.get(key) or "").strip()
        if not value:
            error(key, f"{label} is required.")
            continue
        try:
            number = float(value)
            if number < minimum or (minimum == 0 and number <= 0):
                raise ValueError
        except ValueError:
            error(key, f"{label} must be {'greater than zero' if minimum == 0 else f'at least {minimum:g}'}.")

    epochs = str(settings.get("max_train_epochs") or "").strip()
    steps = str(settings.get("max_train_steps") or "").strip()
    if not settings.get("use_staged_training") and not _positive_integer(epochs) and not _positive_integer(steps):
        error("max_train_epochs", "Set a positive epoch or step limit.")

    if settings.get("dop_enabled"):
        if mode not in {"Krea 2", "Flux.2 Klein"}:
            error("dop_enabled", "DOP is supported only for Krea 2 and FLUX.2 Klein.")
        try:
            from musubi_tuner.training.dop import validate_dop_config

            dop_strength = float(settings.get("dop_loss_weight") or 0)
            validate_dop_config(
                settings.get("dop_trigger_word", ""),
                settings.get("dop_class_word", ""),
                dop_strength,
            )
            if dop_strength <= 0:
                raise ValueError("DOP preservation strength must be greater than zero.")
        except (TypeError, ValueError) as exc:
            error("dop_enabled", f"DOP configuration: {exc}")

    starting_point = settings.get("starting_point_mode")
    resume = "" if starting_point in {"new", "weights"} else str(settings.get("resume_path") or "").strip()
    weights = "" if starting_point in {"new", "state"} else str(settings.get("network_weights") or "").strip()
    if resume and weights:
        error("starting_point_mode", "Saved state and LoRA-weight continuation are mutually exclusive.")
    if resume and not Path(resume).is_dir():
        error("resume_path", f"Saved state folder does not exist: {resume}")
    if weights and not Path(weights).is_file():
        error("network_weights", f"Continuation LoRA does not exist: {weights}")
    if settings.get("resume_exact_position"):
        valid, missing = validate_accelerate_state(resume, exact_position=True)
        if not valid:
            error("resume_path", "Exact recovery state is incomplete: " + ", ".join(missing))

    if settings.get("use_staged_training"):
        try:
            from modern_gui.stages import validate_stage_plan

            validate_stage_plan(settings)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            error("staged_training_config", str(exc))
        if str(settings.get("resume_path") or "").strip():
            error(
                "resume_path",
                "Saved-state recovery cannot be combined with a staged plan. "
                "Recover the interrupted run normally, or continue the resulting LoRA in a new staged plan.",
            )

    for key, label in (
        ("sample_every_n_epochs", "Sample epoch cadence"),
        ("sample_every_n_steps", "Sample step cadence"),
    ):
        value = str(settings.get(key) or "").strip()
        if value not in {"", "0"} and not _positive_integer(value):
            error(key, f"{label} must be a positive whole number, 0, or blank.")

    for message in validate_sample_prompts(settings):
        error("sample_prompts_data", message)
    cadence_enabled = any(
        (
            _positive_integer(str(settings.get("sample_every_n_epochs") or "").strip()),
            _positive_integer(str(settings.get("sample_every_n_steps") or "").strip()),
            bool(settings.get("sample_at_first")),
        )
    )
    if cadence_enabled and "sample_prompts_data" in settings and not enabled_sample_prompts(settings):
        warning(
            "sample_prompts_data",
            "Sampling is scheduled, but no sample prompt is included. Training will run without samples.",
        )

    output_dir = str(settings.get("output_dir") or "").strip()
    if output_dir:
        parent = Path(output_dir).expanduser()
        existing = parent if parent.exists() else parent.parent
        if not existing.exists():
            warning("output_dir", f"Output parent does not currently exist: {existing}")

    return {"errors": errors, "warnings": warnings}


def require_valid_training_settings(settings: dict[str, Any]) -> None:
    result = validate_training_settings(settings)
    if result["errors"]:
        raise ValueError("Training settings are not ready: " + " ".join(item["message"] for item in result["errors"]))


def _positive_integer(value: str) -> bool:
    return value.isdigit() and int(value) > 0
