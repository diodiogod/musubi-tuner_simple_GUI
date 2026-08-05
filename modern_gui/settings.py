from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LAST_SETTINGS = ROOT / "last_settings.json"
MINIMAX_H3_MODE = "MiniMax H3 (Experimental)"
MODES = ["Wan 2.2", "Flux.2 Klein", "Flux.2 Dev", "Krea 2", MINIMAX_H3_MODE]
STRUCTURED_KEYS = {"sample_prompts_data", "staged_training_config", "face_refinement_config"}
MINIMAX_H3_DEFAULTS = {
    "minimax_h3_dit_model": "",
    "minimax_h3_text_encoder": "",
    "minimax_h3_tokenizer": "Qwen/Qwen3-VL-32B-Instruct",
    "minimax_h3_convrot_bwd_mode": "bf16",
    "minimax_h3_text_encoder_load_mode": "auto",
    "minimax_h3_text_cache_dtype": "bfloat16",
    "minimax_h3_preview_decode_min_free_gb": "9.0",
    "minimax_h3_depth_vae_device": "training",
    "minimax_h3_keep_depth_vae_on_device": False,
    "minimax_h3_depth_every_n_steps": "1",
}

MINIMAX_H3_SHARED_REGULARIZATION_KEYS = {
    "krea2_generalization_preset",
    "krea2_weight_noise_sigma",
    "krea2_weight_noise_mode",
    "krea2_weight_noise_bound_norm",
    "krea2_depth_anchor_weight",
    "krea2_depth_anchor_model",
    "krea2_depth_anchor_input_size",
    "krea2_depth_anchor_gradient_weight",
    "krea2_depth_anchor_grad_checkpoint",
    "krea2_keep_depth_helpers_on_gpu",
}

FIELD_LABELS = {
    "minimax_h3_dit_model": "Pruned ConvRot INT8 DiT (Required)",
    "minimax_h3_text_encoder": "Compact Qwen3-VL-32B Text Encoder",
    "recache_latents": "Rebuild Image/Latent Cache",
    "recache_text": "Rebuild Caption/Text Cache",
}

SECTION_TITLES = {
    "essentials": "Essentials",
    "models": "Models",
    "starting_point": "Starting point",
    "network": "Network",
    "optimization": "Optimization",
    "timesteps": "Timesteps",
    "regularization": "Regularization",
    "runtime": "Memory & runtime",
    "logging": "Logging",
    "sampling": "Sampling",
    "staging": "Training plan",
    "utilities": "Utilities",
    "other": "Other",
}

CHOICES = {
    "training_mode": MODES,
    "starting_point_mode": ["new", "weights", "state"],
    "network_type": ["LoRA", "LoHa", "LoKr"],
    "optimizer_type": ["adamw8bit", "AdamW", "Adafactor", "Prodigy", "DAdaptation", "Lion"],
    "lr_scheduler": ["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear", "polynomial", "rex"],
    "mixed_precision": ["no", "fp16", "bf16"],
    "attention_mechanism": ["sdpa", "xformers", "flash", "sageattn"],
    "timestep_sampling": ["sigma", "uniform", "sigmoid", "shift", "flux_shift", "krea2_shift"],
    "compile_backend": ["inductor", "eager", "aot_eager", "cudagraphs"],
    "compile_mode": ["default", "reduce-overhead", "max-autotune"],
    "compile_dynamic": ["auto", "true", "false"],
    "log_with": ["none", "tensorboard", "wandb"],
    "krea2_weight_noise_mode": ["relative", "absolute"],
    "krea2_generalization_preset": ["Off (Baseline)", "Weight Noise Only", "Balanced Experimental"],
    "minimax_h3_convrot_bwd_mode": ["bf16", "int8"],
    "minimax_h3_text_cache_dtype": ["bfloat16", "float32"],
    "minimax_h3_depth_vae_device": ["training", "secondary"],
    "appearance_mode": ["Dark", "Light"],
}

PATH_KEYS = {
    "dataset_config",
    "project_root",
    "output_dir",
    "network_weights",
    "resume_path",
    "dit_high_noise",
    "dit_low_noise",
    "clip_model",
    "t5_model",
    "flux2_dit_model",
    "flux2_text_encoder",
    "krea2_dit_model",
    "krea2_text_encoder",
    "krea2_turbo_dit",
    "krea2_projector_diff",
    "minimax_h3_dit_model",
    "minimax_h3_text_encoder",
    "vae_model",
    "logging_dir",
    "convert_lora_path",
    "convert_output_dir",
}

TEXTAREA_KEYS = {"training_comment", "optimizer_args"}

MODE_RULES = {
    "is_i2v": ["Wan 2.2"],
    "train_high_noise": ["Wan 2.2"],
    "dit_high_noise": ["Wan 2.2"],
    "min_timestep_high": ["Wan 2.2"],
    "max_timestep_high": ["Wan 2.2"],
    "train_low_noise": ["Wan 2.2"],
    "dit_low_noise": ["Wan 2.2"],
    "min_timestep_low": ["Wan 2.2"],
    "max_timestep_low": ["Wan 2.2"],
    "clip_model": ["Wan 2.2"],
    "t5_model": ["Wan 2.2"],
    "force_v2_1_time_embedding": ["Wan 2.2"],
    "offload_inactive_dit": ["Wan 2.2"],
    "network_dim_high": ["Wan 2.2"],
    "network_alpha_high": ["Wan 2.2"],
}


def load_settings() -> dict[str, Any]:
    if not LAST_SETTINGS.exists():
        return {}
    payload = json.loads(LAST_SETTINGS.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    for key, value in MINIMAX_H3_DEFAULTS.items():
        payload.setdefault(key, value)
    return payload


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a JSON object.")
    temporary = LAST_SETTINGS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(LAST_SETTINGS)
    return settings


def _humanize(key: str) -> str:
    names = {
        "dit": "DiT",
        "vae": "VAE",
        "t5": "T5",
        "fp8": "FP8",
        "dop": "DOP",
        "lokr": "LoKr",
        "lr": "LR",
        "cfg": "CFG",
        "gpu": "GPU",
    }
    return " ".join(names.get(piece, piece.capitalize()) for piece in key.split("_"))


def _section_for(key: str) -> str:
    if key in {"training_mode", "dataset_config", "project_root", "output_dir", "output_name", "appearance_mode"}:
        return "essentials"
    if key in PATH_KEYS and key not in {"dataset_config", "project_root", "output_dir", "network_weights", "resume_path"}:
        return "models"
    if key in {"starting_point_mode", "network_weights", "resume_path"}:
        return "starting_point"
    if key.startswith("network_") or key in {"network_type", "lokr_factor"}:
        return "network"
    if key.startswith(("learning_", "optimizer", "lr_", "max_train", "save_every", "gradient_accumulation", "max_grad", "seed")):
        return "optimization"
    if "timestep" in key or key in {"discrete_flow_shift", "preserve_distribution_shape", "train_high_noise", "train_low_noise"}:
        return "timesteps"
    if key.startswith(("dop_", "krea2_weight_", "krea2_depth_", "krea2_generalization", "krea2_keep_", "minimax_h3_depth_", "minimax_h3_keep_depth_")):
        return "regularization"
    if key.startswith(("sample_",)):
        return "sampling"
    if key.startswith(("staged_", "use_staged")):
        return "staging"
    if key.startswith(("log_", "logging_", "auto_training", "training_comment")):
        return "logging"
    if key.startswith("convert_"):
        return "utilities"
    if key in {
        "mixed_precision",
        "gradient_checkpointing",
        "persistent_data_loader_workers",
        "max_data_loader_n_workers",
        "blocks_to_swap",
        "compile",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_fullgraph",
        "compile_cache_size_limit",
        "attention_mechanism",
        "fp8_base",
        "fp8_scaled",
        "fp8_t5",
        "fp8_llm",
        "minimax_h3_convrot_bwd_mode",
        "save_state",
        "rename_final_artifacts_to_epoch",
        "recache_latents",
        "recache_text",
        "offload_inactive_dit",
        "force_v2_1_time_embedding",
    }:
        return "runtime"
    return "models" if re.search(r"(model|dit|text_encoder|projector|vae|clip|t5)", key) else "other"


def _field_for(key: str, value: Any) -> dict[str, Any]:
    if key in CHOICES:
        field_type = "select"
    elif isinstance(value, bool):
        field_type = "boolean"
    elif key in TEXTAREA_KEYS:
        field_type = "textarea"
    elif key in PATH_KEYS:
        field_type = "path"
    else:
        field_type = "text"
    field = {"key": key, "label": FIELD_LABELS.get(key, _humanize(key)), "type": field_type}
    if key in CHOICES:
        field["options"] = CHOICES[key]
        field["allow_custom"] = key not in {"training_mode", "appearance_mode"}
    modes = MODE_RULES.get(key)
    if key.startswith("flux2_") or key == "fp8_text_encoder":
        modes = ["Flux.2 Klein", "Flux.2 Dev"]
    elif key in MINIMAX_H3_SHARED_REGULARIZATION_KEYS:
        modes = ["Krea 2", MINIMAX_H3_MODE]
    elif key.startswith("krea2_") or key.startswith("dop_"):
        modes = ["Krea 2"] if key.startswith("krea2_") else ["Krea 2", "Flux.2 Klein", MINIMAX_H3_MODE]
    elif key.startswith("minimax_h3_"):
        modes = [MINIMAX_H3_MODE]
    if key in {"network_type", "mixed_precision", "compile", "fp8_base", "fp8_scaled"}:
        field["disabled_modes"] = [MINIMAX_H3_MODE]
    if modes:
        field["modes"] = modes
    return field


def settings_schema(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings if settings is not None else load_settings()
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in SECTION_TITLES}
    for key, value in settings.items():
        if key in STRUCTURED_KEYS:
            continue
        grouped[_section_for(key)].append(_field_for(key, value))
    sections = [
        {"id": section_id, "title": SECTION_TITLES[section_id], "fields": fields}
        for section_id, fields in grouped.items()
        if fields
    ]
    return {
        "modes": MODES,
        "sections": sections,
        "structured_keys": sorted(STRUCTURED_KEYS),
        "field_count": sum(len(section["fields"]) for section in sections),
    }
