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
    elif mode == "MiniMax H3 (Experimental)":
        require("minimax_h3_dit_model", "MiniMax H3 pruned ConvRot INT8 DiT")
        cadence_enabled = any(
            (
                _positive_number(str(settings.get("sample_every_n_epochs") or "").strip()),
                _positive_integer(str(settings.get("sample_every_n_steps") or "").strip()),
                bool(settings.get("sample_at_first")),
            )
        )
        has_face_stage = any(stage.get("type") == "face_refinement" for stage in configured_stages)
        if settings.get("recache_text") or cadence_enabled or has_face_stage:
            require("minimax_h3_text_encoder", "MiniMax H3 compact Qwen3-VL text encoder")
        if settings.get("network_type", "LoRA") != "LoRA":
            error("network_type", "Experimental MiniMax H3 currently supports LoRA only.")
        if settings.get("fp8_base") or settings.get("fp8_scaled"):
            error("fp8_base", "MiniMax H3 already loads the pruned ConvRot INT8 base directly; FP8 flags must be disabled.")
        if settings.get("compile"):
            error("compile", "Torch Compile is not supported by the experimental MiniMax H3 ConvRot path.")
        if settings.get("mixed_precision") != "bf16":
            error("mixed_precision", "Experimental MiniMax H3 requires BF16 mixed precision.")
        if not settings.get("gradient_checkpointing"):
            error("gradient_checkpointing", "MiniMax H3 H2D-only block swapping requires gradient checkpointing.")
        try:
            blocks_to_swap = int(str(settings.get("blocks_to_swap") or "0").strip())
        except ValueError:
            blocks_to_swap = 0
        if not 1 <= blocks_to_swap <= 48:
            error("blocks_to_swap", "Use 1–48 swapped blocks for MiniMax H3; 30 is the conservative 24 GB default.")
        warning(
            "training_mode",
            "Experimental image-only path: direct pruned ConvRot INT8 base and batch size 1. A 1024px rank-16 two-epoch run and its LoRA were validated on a 24 GB RTX 4090; previews and advanced regularizers remain experimental, so start them with a short run.",
        )
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
        if mode not in {"Krea 2", "Flux.2 Klein", "MiniMax H3 (Experimental)"}:
            error("dop_enabled", "DOP is supported only for Krea 2, FLUX.2 Klein, and experimental MiniMax H3.")
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

    if mode in {"Krea 2", "MiniMax H3 (Experimental)"}:
        preset = str(settings.get("krea2_generalization_preset") or "").strip()
        try:
            noise_strength = float(settings.get("krea2_weight_noise_sigma") or 0)
            depth_strength = float(settings.get("krea2_depth_anchor_weight") or 0)
        except (TypeError, ValueError):
            noise_strength = depth_strength = 0.0
        if preset == "Off (Baseline)" and (noise_strength > 0 or depth_strength > 0):
            error(
                "krea2_generalization_preset",
                "Generalization preset is Off, but weight noise or depth anchoring is still nonzero. "
                "Press Apply selected preset to set both values to zero before launching.",
            )
        if mode == "MiniMax H3 (Experimental)" and depth_strength > 0:
            vae_device = str(settings.get("minimax_h3_depth_vae_device") or "training").strip().lower()
            if vae_device not in {"training", "secondary"} and not vae_device.startswith("cuda:"):
                error(
                    "minimax_h3_depth_vae_device",
                    "Depth VAE device must be training, secondary, or an advanced logical CUDA device such as cuda:1.",
                )
            try:
                cadence = int(str(settings.get("minimax_h3_depth_every_n_steps") or "1"))
                if cadence <= 0:
                    raise ValueError
            except ValueError:
                error("minimax_h3_depth_every_n_steps", "Depth cadence must be a positive whole number.")
            if vae_device == "secondary" and not settings.get("minimax_h3_keep_depth_vae_on_device"):
                warning(
                    "minimax_h3_keep_depth_vae_on_device",
                    "The secondary VAE is not kept loaded, so approximately 5 GB of weights will move repeatedly. "
                    "Enable Keep VAE on Selected GPU unless that GPU must be shared.",
                )

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
        valid = _positive_number(value) if key == "sample_every_n_epochs" else _positive_integer(value)
        if value not in {"", "0"} and not valid:
            expected = "a positive number" if key == "sample_every_n_epochs" else "a positive whole number"
            error(key, f"{label} must be {expected}, 0, or blank.")

    for key, label in (
        ("save_every_n_epochs", "Save epoch cadence"),
        ("save_every_n_steps", "Save step cadence"),
    ):
        value = str(settings.get(key) or "").strip()
        if value not in {"", "0"} and not _positive_integer(value):
            error(key, f"{label} must be a positive whole number, 0, or blank.")

    for message in validate_sample_prompts(settings):
        error("sample_prompts_data", message)
    if mode == "MiniMax H3 (Experimental)":
        for index, prompt in enumerate(enabled_sample_prompts(settings), start=1):
            label = f"Sample prompt {index}"
            if str(prompt.get("neg") or "").strip():
                error("sample_prompts_data", f"{label} cannot use a negative prompt with MiniMax H3.")
            try:
                guidance = float(prompt.get("guidance") or 1.0)
                cfg_scale = float(prompt.get("cfg_scale") or 1.0)
                frames = int(prompt.get("frames") or 1)
                width = int(prompt.get("width") or 768)
                height = int(prompt.get("height") or 768)
            except (TypeError, ValueError):
                continue  # The generic prompt validator reports the malformed field.
            if guidance != 1.0 or cfg_scale != 1.0:
                error("sample_prompts_data", f"{label} must keep MiniMax H3 guidance and CFG at 1.0.")
            if frames != 1 and (frames < 5 or (frames - 5) % 17):
                error("sample_prompts_data", f"{label} frames must be 1 or MiniMax H3 video lengths 5, 22, 39, ...")
            elif frames != 1:
                five_frame_scheduled = (
                    str(settings.get("minimax_h3_training_preview_mode") or "").strip().lower()
                    in {"five_frame", "five-frame video (experimental)"}
                )
                scheduled_description = "five frames" if five_frame_scheduled else "one frame"
                warning(
                    "sample_prompts_data",
                    f"{label} keeps {frames} frames for standalone Preview; scheduled training samples use {scheduled_description}.",
                )
            if width % 32 or height % 32:
                error("sample_prompts_data", f"{label} MiniMax H3 width and height must be multiples of 32.")
    cadence_enabled = any(
        (
            _positive_number(str(settings.get("sample_every_n_epochs") or "").strip()),
            _positive_integer(str(settings.get("sample_every_n_steps") or "").strip()),
            bool(settings.get("sample_at_first")),
        )
    )
    if mode == "MiniMax H3 (Experimental)":
        preview_mode = str(settings.get("minimax_h3_training_preview_mode") or "One frame (safe)").strip().lower()
        valid_preview_modes = {
            "one_frame", "five_frame", "one frame (safe)", "five-frame video (experimental)",
        }
        if preview_mode not in valid_preview_modes:
            error("minimax_h3_training_preview_mode", "Choose the safe one-frame or experimental five-frame scheduled preview mode.")
        elif cadence_enabled and preview_mode in {"five_frame", "five-frame video (experimental)"}:
            warning(
                "minimax_h3_training_preview_mode",
                "Five-frame scheduled MiniMax previews are slower and may temporarily use more VRAM. Start with a conservative sample cadence.",
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


def _positive_number(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number > 0 and number == number and number not in {float("inf"), float("-inf")}
