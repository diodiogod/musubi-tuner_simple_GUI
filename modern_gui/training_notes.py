from __future__ import annotations

import tomllib
import math
from pathlib import Path
from typing import Any

from modern_gui.stages import stage_label


def _enabled_number(settings: dict[str, Any], key: str) -> str | None:
    try:
        value = float(settings.get(key) or 0)
    except (TypeError, ValueError):
        return None
    return f"{value:g}" if math.isfinite(value) and value > 0 else None


def _stage_limit(stage: dict[str, Any]) -> str:
    steps = str(stage.get("steps") or "").strip()
    if steps:
        return f"{steps} steps"
    return f"{str(stage.get('epochs') or '?').strip()} epochs"


def _dataset_context(value: Any, text: str | None = None) -> tuple[str, list[int]]:
    if not str(value or "").strip():
        return "", []
    source = Path(str(value or "")).expanduser()
    label = source.stem
    try:
        payload = tomllib.loads(text if text is not None else source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError):
        return label, []
    general = payload.get("general", {})
    resolutions: list[str] = []
    batches = set()
    for item in payload.get("datasets", []):
        resolution = item.get("resolution", general.get("resolution"))
        if resolution is not None:
            resolution = "×".join(str(part) for part in resolution) if isinstance(resolution, (list, tuple)) else str(resolution)
            if resolution not in resolutions:
                resolutions.append(resolution)
        batch = item.get("batch_size", general.get("batch_size", 1))
        if isinstance(batch, int) and not isinstance(batch, bool) and batch > 0:
            batches.add(batch)
    sizes = "/".join(resolutions[:4]) + (f"/+{len(resolutions)-4}" if len(resolutions) > 4 else "")
    return (f"{label} ({sizes})" if sizes else label), sorted(batches)


def _dataset_note_label(value: Any) -> str:
    return _dataset_context(value)[0]


def _compact_list(values: list[int]) -> str:
    return "/".join(map(str, values[:4])) + (f"/+{len(values)-4}" if len(values) > 4 else "")


def _h3_features(settings: dict[str, Any]) -> list[str]:
    from backends.minimax_h3 import quality_protection_components, is_multimodal, teacher_condition_value, assistant_source
    native = is_multimodal(settings)
    parts = [f"media={settings.get('minimax_h3_multimodal_task', 't2va')}/{settings.get('minimax_h3_training_target', 'Video + audio')}" if native else "media=still images"]
    protection = quality_protection_components(settings)
    teacher = settings.get("minimax_h3_teacher_matching")
    if teacher:
        parts.append(f"reference-learning={teacher_condition_value(settings.get('minimax_h3_teacher_conditions'))} σ≤{settings.get('minimax_h3_teacher_condition_sigma_max') or '0.75'} direct={settings.get('minimax_h3_teacher_direct_loss_weight') or '0'}")
    else:
        if protection["dynamic"]:
            every = "1" if native else settings.get("minimax_h3_dynamic_sigma_every_n_steps") or "1"
            parts.append(f"Dynamic Sigma {settings.get('minimax_h3_guidance_distillation_scale') or '4'} σ≥{settings.get('minimax_h3_guidance_distillation_sigma_min') or '0.15'} every={every}")
        if protection["assistant"]:
            parts.append(f"assistant={Path(assistant_source(settings)).name}")
    if protection["base"] and not native:
        reference = settings.get("minimax_h3_base_preservation_reference") or "Base + assistant"
        if not protection["assistant"]:
            reference = "Base only"
        parts.append(f"drift {settings.get('minimax_h3_base_preservation_loss_weight') or '0.05'} every={settings.get('minimax_h3_base_preservation_every_n_steps') or '10'} ({reference})")
    return parts


def training_settings_summary(settings: dict[str, Any], *, dataset_text: str | None = None) -> str:
    from backends.regularization_switches import effective_settings
    settings = effective_settings(settings)
    mode = str(settings.get("training_mode") or "Training")
    from backends.minimax_h3 import is_multimodal
    native_h3 = mode == "MiniMax H3 (Experimental)" and is_multimodal(settings)
    dataset, batches = _dataset_context(settings.get("dataset_config"), dataset_text)
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
    if network == "LoKr" and str(settings.get("lokr_factor") or "").strip():
        network_text += f" factor {settings['lokr_factor']}"
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
        if epochs:
            parts.append(f"{epochs} epochs")
        elif steps:
            parts.append(f"{steps} steps")
        if dataset:
            parts.append(f"data={dataset}")

    if batches:
        parts.append(f"batch={_compact_list(batches)}")
    accumulation = str(settings.get("gradient_accumulation_steps") or "1")
    parts.append(f"accum={accumulation}")
    try:
        count = int(accumulation)
        if batches and count > 0:
            effective = f"{batches[0]*count}–{batches[-1]*count}" if len(batches) > 1 else str(batches[0]*count)
            parts.append(f"effective-batch≈{effective}/GPU")
    except ValueError:
        pass

    learning_rate = str(settings.get("learning_rate") or "").strip()
    if learning_rate:
        parts.append(f"lr={learning_rate}")
    optimizer = str(settings.get("optimizer_type") or "").strip()
    if optimizer:
        parts.append(f"opt={optimizer}")
    model_key = {"Krea 2": "krea2_dit_model", "MOSS-Image (Experimental)": "moss_dit_model",
                 "MiniMax H3 (Experimental)": "minimax_h3_dit_model", "Flux.2 Klein": "flux2_dit_model",
                 "Flux.2 Dev": "flux2_dit_model"}.get(mode)
    if model_key and settings.get(model_key):
        parts.append(f"base={Path(str(settings[model_key])).name}")
    for key, label in (("lr_scheduler", "sched"), ("lr_warmup_steps", "warmup"), ("max_grad_norm", "clip"), ("seed", "seed"), ("mixed_precision", "precision")):
        value = str(settings.get(key) if settings.get(key) is not None else "").strip()
        if value:
            parts.append(f"{label}={value}")
    optimizer_args = str(settings.get("optimizer_args") or "").strip()
    if optimizer_args:
        parts.append(f"opt-args={optimizer_args}")
    if mode == "MOSS-Image (Experimental)" and optimizer.lower() == "automagic3":
        parts.append(f"LR-bounds={settings.get('moss_automagic3_min_lr') or '0.00005'}..{settings.get('moss_automagic3_max_lr') or '0.001'}")
    if native_h3:
        parts.append("noise=linked uniform")
    elif mode != "MOSS-Image (Experimental)":
        for key, label in (("timestep_sampling", "noise"), ("discrete_flow_shift", "shift")):
            if str(settings.get(key) or "").strip():
                parts.append(f"{label}={settings[key]}")

    if mode == "MiniMax H3 (Experimental)" and settings.get("minimax_h3_foundation_lora_enabled"):
        foundation = str(settings.get("minimax_h3_foundation_lora") or "").strip()
        strength = str(settings.get("minimax_h3_foundation_lora_multiplier") or "1.0").strip()
        if foundation:
            parts.append(f"foundation={Path(foundation).name}@{strength}")

    if settings.get("dop_enabled") and mode in {"Krea 2", "Flux.2 Klein", "MOSS-Image (Experimental)", "MiniMax H3 (Experimental)"} and not native_h3:
        strength = _enabled_number(settings, "dop_loss_weight")
        class_word = str(settings.get("dop_class_word") or "").strip()
        text = f"DOP {strength or '?'}"
        if class_word:
            text += f" ({class_word})"
        parts.append(text)

    if mode == "MiniMax H3 (Experimental)":
        parts.extend(_h3_features(settings))
    if mode in {"Krea 2", "MiniMax H3 (Experimental)"} and not native_h3:
        depth = _enabled_number(settings, "krea2_depth_anchor_weight")
        if depth:
            text = f"depth {depth}@{settings.get('krea2_depth_anchor_input_size') or 518}"
            keep = settings.get("minimax_h3_keep_depth_vae_on_device") if mode == "MiniMax H3 (Experimental)" else settings.get("krea2_keep_depth_helpers_on_gpu")
            text += " GPU" if keep else " offload"
            device = settings.get("minimax_h3_depth_vae_device" if mode == "MiniMax H3 (Experimental)" else "krea2_depth_vae_device")
            if device and device != "training":
                text += f" VAE={device}"
            parts.append(text)
        noise = _enabled_number(settings, "krea2_weight_noise_sigma")
        if noise:
            parts.append(
                f"weight-noise {noise} {settings.get('krea2_weight_noise_mode') or 'relative'}"
            )
        projector = str(settings.get("krea2_projector_diff") or "").strip()
        if projector and mode == "Krea 2":
            strength = str(settings.get("krea2_projector_diff_strength") or "1").strip()
            parts.append(f"projector={Path(projector).name}@{strength}")
    if mode == "Flux.2 Klein" and settings.get("flux2_reference_guided"):
        from backends.flux2 import reference_condition_value
        parts.append(f"reference-learning={reference_condition_value(settings.get('flux2_reference_conditions'))} σ≤{settings.get('flux2_reference_sigma_max') or '0.75'} direct={settings.get('flux2_reference_direct_loss_weight') or '0'}")

    if str(settings.get("training_mode")) == "MOSS-Image (Experimental)":
        from backends.moss_settings import timestep_schedule
        parts.append(f"noise={timestep_schedule(settings)}")
        if timestep_schedule(settings) == "MOSS pretraining (default)":
            share = str(settings.get("moss_clean_end_share") or "").strip()
            if share:
                parts.append(f"clean-end={share}")
        if settings.get("moss_repa_enabled"):
            parts.append(f"REPA {settings.get('moss_repa_weight', '0.5')}@layer{settings.get('moss_repa_block', '8')}")
        if settings.get("moss_ema_enabled"):
            parts.append(f"EMA {settings.get('moss_ema_decay', '0.98')}")
        depth = _enabled_number(settings, "moss_depth_anchor_weight")
        if depth:
            text = f"depth {depth}@{settings.get('moss_depth_anchor_input_size') or 518}"
            text += " GPU" if settings.get("moss_keep_depth_helpers_on_gpu") else " offload"
            device = settings.get("moss_depth_anchor_vae_device")
            if device and device != "training":
                text += f" VAE={device}"
            parts.append(text)
        noise = _enabled_number(settings, "moss_weight_noise_sigma")
        if noise:
            parts.append(f"weight-noise {noise} {settings.get('moss_weight_noise_mode') or 'relative'}")
        if settings.get("moss_lora_target_preset") == "Attention only":
            parts.append("targets=attention-only")
        if settings.get("moss_fast_lora") and network == "LoRA":
            parts.append("BF16-LoRA")
        if settings.get("moss_unpadded_batches"):
            parts.append("unpadded")

    if settings.get("gradient_checkpointing"):
        parts.append("grad-ckpt" + (" CPU" if settings.get("gradient_checkpointing_cpu_offload") else ""))
    if settings.get("compile") and mode != "MiniMax H3 (Experimental)":
        parts.append("compile" + (f"/{settings.get('moss_compile_checkpoint') or 'inside'}" if mode == "MOSS-Image (Experimental)" else ""))
    if settings.get("fp8_base") and mode != "MiniMax H3 (Experimental)":
        parts.append("FP8-scaled" if settings.get("fp8_scaled") else "FP8")

    blocks = str(settings.get("blocks_to_swap") or "").strip()
    if settings.get("training_mode") == "MiniMax H3 (Experimental)" and settings.get("minimax_h3_block_memory_mode") == "Automatic (experimental)":
        parts.append("swap=automatic")
    elif blocks:
        parts.append(f"swap={blocks}")
    return "Settings: " + "; ".join(parts)


def effective_training_comment(settings: dict[str, Any]) -> str:
    custom = str(settings.get("training_comment") or "").strip()
    if not settings.get("auto_training_settings_summary", True):
        return custom
    generated = training_settings_summary(settings)
    return "\n\n".join(part for part in (custom, generated) if part)
