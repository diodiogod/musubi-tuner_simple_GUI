"""Command construction for compact still-image and official multimodal MiniMax-H3 training."""

from pathlib import Path
import math

from backends._common import setting_or_default

DEFAULT_H3_TRAINING_ASSISTANT = (
    "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v1.safetensors"
)


def add_automatic_swap_args(cmd, settings, add_arg):
    validate_automatic_swap_settings(settings)
    if settings.get("minimax_h3_block_memory_mode") != "Automatic (experimental)":
        return
    add_arg(cmd, "--auto_block_swap", True)
    for setting, flag, default in (
        ("minimax_h3_auto_swap_reserve_gb", "--auto_swap_reserve_gb", "2.0"),
        ("minimax_h3_auto_swap_min_blocks", "--auto_swap_min_blocks", "2"),
        ("minimax_h3_auto_swap_max_blocks", "--auto_swap_max_blocks", "48"),
    ):
        add_arg(cmd, flag, setting_or_default(settings, setting, default))


def validate_automatic_swap_settings(settings):
    if settings.get("minimax_h3_block_memory_mode") != "Automatic (experimental)":
        return
    try:
        reserve = float(setting_or_default(settings, "minimax_h3_auto_swap_reserve_gb", "2.0"))
        minimum = int(setting_or_default(settings, "minimax_h3_auto_swap_min_blocks", "2"))
        maximum = int(setting_or_default(settings, "minimax_h3_auto_swap_max_blocks", "48"))
        if not math.isfinite(reserve) or reserve < 0 or not 2 <= minimum <= maximum <= 48:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("Automatic H3 memory needs a finite nonnegative safety margin and 2 <= minimum <= maximum <= 48 blocks") from None
    if settings.get("compile"):
        raise ValueError("Automatic H3 memory is not yet compatible with Torch Compile; disable Compile or use Fixed blocks")


def teacher_condition_value(value):
    """Map friendly GUI labels and legacy saved values to the trainer contract."""
    normalized = str(value or "ref").strip().lower()
    if normalized.startswith("same training item") or normalized == "ref":
        return "ref"
    if normalized.startswith("other pictures") or normalized == "subject_ref":
        return "subject_ref"
    if normalized.startswith("first and last") or normalized == "first,last":
        return "first,last"
    return normalized


def quality_protection_method(settings):
    """Return the CLI value while accepting old saved boolean recipes."""
    value = str(settings.get("minimax_h3_quality_protection_method") or "").strip().lower()
    aliases = {
        "dynamic sigma (recommended)": "dynamic",
        "ostris assistant (alpha)": "assistant",
        "ostris assistant (v1)": "assistant",
        "assistant + base preservation (alpha)": "assistant_preservation",
        "assistant + base preservation (v1)": "assistant_preservation",
        "off": "off",
    }
    if value in {"dynamic", "assistant", "assistant_preservation", "off"}:
        return value
    if value in aliases:
        return aliases[value]
    return "dynamic" if settings.get("minimax_h3_guidance_distillation_protection", True) else "off"


def quality_protection_components(settings):
    """Resolve independent controls, migrating recipes from the old method selector."""
    if "minimax_h3_training_assistant_enabled" in settings:
        return {
            "assistant": bool(settings.get("minimax_h3_training_assistant_enabled")),
            "dynamic": bool(settings.get("minimax_h3_dynamic_sigma_enabled")),
            "base": bool(settings.get("minimax_h3_base_preservation_enabled")),
        }
    method = quality_protection_method(settings)
    return {
        "assistant": method in {"assistant", "assistant_preservation"},
        "dynamic": method == "dynamic",
        "base": method == "assistant_preservation",
    }


def is_multimodal(settings):
    return str(settings.get("minimax_h3_training_workflow") or "Still images · compact ConvRot").startswith("Video")


def is_mixed_one_frame(settings):
    return str(settings.get("minimax_h3_training_workflow") or "").startswith("Video + images")


def _guidance_cache_path(settings):
    configured = str(settings.get("minimax_h3_guidance_uncond_cache") or "").strip()
    if configured:
        return configured
    output_dir, output_name = build_output_dir(settings)
    return str(Path(output_dir) / f"{output_name}_h3_uncond.safetensors")

from backends._common import (
    add_arg,
    build_attention_arg,
    build_common_train_args,
    build_output_dir,
    build_sample_args,
    build_dop_cache_args,
    build_dop_train_args,
)


def build_commands(settings):
    if settings.get("network_type", "LoRA") != "LoRA":
        raise ValueError("Experimental MiniMax-H3 currently supports LoRA only")
    if is_multimodal(settings):
        return _build_multimodal_commands(settings)
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        "1",
        "--num_cpu_threads_per_process",
        "1",
        "src/musubi_tuner/minimax_h3_image_train_network.py",
    ]
    add_arg(cmd, "--mixed_precision", settings.get("mixed_precision"))
    add_arg(cmd, "--dit", settings.get("minimax_h3_dit_model"), is_path=True)
    add_arg(cmd, "--vae", settings.get("vae_model"), is_path=True)
    add_arg(cmd, "--dataset_config", settings.get("dataset_config"), is_path=True)
    add_arg(cmd, "--text_encoder", settings.get("minimax_h3_text_encoder"), is_path=True)
    add_arg(cmd, "--tokenizer", settings.get("minimax_h3_tokenizer"))
    add_arg(cmd, "--text_encoder_load_mode", settings.get("minimax_h3_text_encoder_load_mode") or "auto")
    add_arg(cmd, "--text_encoder_blocks_to_swap", setting_or_default(settings, "minimax_h3_text_encoder_blocks_to_swap", "50"))
    add_arg(
        cmd,
        "--minimax_h3_preview_decode_min_free_gb",
        settings.get("minimax_h3_preview_decode_min_free_gb") or "9.0",
    )
    add_arg(cmd, "--network_module", "networks.lora_minimax_h3")
    add_arg(cmd, "--network_dim", settings.get("network_dim_low"))
    add_arg(cmd, "--network_alpha", settings.get("network_alpha_low"))
    if settings.get("minimax_h3_foundation_lora_enabled"):
        add_arg(cmd, "--h3_foundation_lora", settings.get("minimax_h3_foundation_lora"), is_path=True)
        add_arg(
            cmd,
            "--h3_foundation_lora_multiplier",
            settings.get("minimax_h3_foundation_lora_multiplier") or "1.0",
        )
    add_arg(cmd, "--blocks_to_swap", settings.get("blocks_to_swap"))
    add_automatic_swap_args(cmd, settings, add_arg)
    add_arg(cmd, "--block_swap_h2d_only", True)
    add_arg(cmd, "--block_swap_ring_size", settings.get("block_swap_ring_size"))
    add_arg(cmd, "--use_pinned_memory_for_block_swap", settings.get("use_pinned_memory_for_block_swap"))
    add_arg(cmd, "--convrot_bwd_mode", settings.get("minimax_h3_convrot_bwd_mode") or "bf16")
    protection = quality_protection_components(settings)
    teacher_matching = bool(settings.get("minimax_h3_teacher_matching"))
    if teacher_matching:
        add_arg(cmd, "--h3_teacher_matching", True)
        add_arg(cmd, "--h3_teacher_conditions", teacher_condition_value(settings.get("minimax_h3_teacher_conditions")))
        add_arg(cmd, "--h3_teacher_condition_sigma_max", settings.get("minimax_h3_teacher_condition_sigma_max") or "0.75")
        add_arg(cmd, "--h3_teacher_direct_loss_weight", settings.get("minimax_h3_teacher_direct_loss_weight") or "0.0")
    add_arg(cmd, "--h3_training_assistant_enabled", protection["assistant"] and not teacher_matching)
    add_arg(cmd, "--h3_guidance_distillation_protection", protection["dynamic"] and not teacher_matching)
    add_arg(cmd, "--h3_dynamic_sigma_every_n_steps", settings.get("minimax_h3_dynamic_sigma_every_n_steps") or "1")
    add_arg(cmd, "--h3_guidance_distillation_scale", settings.get("minimax_h3_guidance_distillation_scale") or "4.0")
    add_arg(cmd, "--h3_guidance_distillation_schedule", settings.get("minimax_h3_guidance_distillation_schedule") or "sigma")
    add_arg(cmd, "--h3_guidance_distillation_sigma_min", settings.get("minimax_h3_guidance_distillation_sigma_min") or "0.15")
    if protection["assistant"]:
        add_arg(
            cmd,
            "--h3_training_assistant",
            settings.get("minimax_h3_training_assistant") or DEFAULT_H3_TRAINING_ASSISTANT,
        )
    add_arg(cmd, "--h3_base_preservation_enabled", protection["base"])
    if protection["base"]:
        add_arg(
            cmd,
            "--h3_base_preservation_loss_weight",
            settings.get("minimax_h3_base_preservation_loss_weight") or "0.05",
        )
        add_arg(
            cmd,
            "--h3_base_preservation_every_n_steps",
            settings.get("minimax_h3_base_preservation_every_n_steps") or "10",
        )
        reference = str(settings.get("minimax_h3_base_preservation_reference") or "Base + assistant").lower()
        if not protection["assistant"]:
            reference = "base only"
        add_arg(cmd, "--h3_base_preservation_reference", "base" if reference == "base only" else "assistant")
    add_arg(cmd, "--weight_noise_sigma", settings.get("krea2_weight_noise_sigma"))
    add_arg(cmd, "--weight_noise_mode", settings.get("krea2_weight_noise_mode"))
    add_arg(cmd, "--weight_noise_bound_norm", settings.get("krea2_weight_noise_bound_norm"))
    add_arg(cmd, "--depth_anchor_weight", settings.get("krea2_depth_anchor_weight"))
    add_arg(cmd, "--depth_anchor_model", settings.get("krea2_depth_anchor_model"))
    add_arg(cmd, "--depth_anchor_input_size", settings.get("krea2_depth_anchor_input_size"))
    add_arg(cmd, "--depth_anchor_gradient_weight", settings.get("krea2_depth_anchor_gradient_weight"))
    if not settings.get("krea2_depth_anchor_grad_checkpoint", True):
        cmd.append("--no-depth_anchor_grad_checkpoint")
    add_arg(cmd, "--keep_depth_helpers_on_gpu", settings.get("krea2_keep_depth_helpers_on_gpu"))
    add_arg(cmd, "--depth_anchor_vae_device", settings.get("minimax_h3_depth_vae_device") or "training")
    add_arg(cmd, "--keep_depth_vae_on_device", settings.get("minimax_h3_keep_depth_vae_on_device"))
    add_arg(cmd, "--depth_anchor_every_n_steps", settings.get("minimax_h3_depth_every_n_steps") or "1")
    build_attention_arg(cmd, settings)
    build_sample_args(cmd, settings)
    build_dop_train_args(cmd, settings)
    build_common_train_args(cmd, settings)

    output_dir, output_name = build_output_dir(settings)
    add_arg(cmd, "--output_dir", output_dir, is_path=True)
    add_arg(cmd, "--output_name", output_name)
    return [cmd]


def _build_multimodal_commands(settings):
    """Build the isolated upstream video/joint-audio trainer command."""
    if is_mixed_one_frame(settings):
        if str(settings.get("minimax_h3_multimodal_task") or "t2va") == "ref2va":
            raise ValueError("Mixed native MiniMax image/video training currently supports T2VA or FL2VA, not Ref2VA")
        if settings.get("minimax_h3_teacher_matching"):
            raise ValueError("Mixed native one-frame training is not yet compatible with teacher matching")
    cmd = [
        "accelerate", "launch", "--num_processes", "1", "--num_cpu_threads_per_process", "1",
        "src/musubi_tuner/minimax_h3_native_train_network.py",
    ]
    add_arg(cmd, "--mixed_precision", settings.get("mixed_precision") or "bf16")
    add_arg(cmd, "--dit", settings.get("minimax_h3_dit_model"), is_path=True)
    add_arg(cmd, "--dataset_config", settings.get("dataset_config"), is_path=True)
    add_arg(cmd, "--task", settings.get("minimax_h3_multimodal_task") or "t2va")
    add_arg(cmd, "--one_frame", is_mixed_one_frame(settings))
    add_arg(cmd, "--video_vae", settings.get("minimax_h3_video_vae"), is_path=True)
    add_arg(cmd, "--audio_vae", settings.get("minimax_h3_audio_vae"), is_path=True)
    add_arg(cmd, "--text_encoder", settings.get("minimax_h3_text_encoder"), is_path=True)
    add_arg(cmd, "--text_encoder_blocks_to_swap", setting_or_default(settings, "minimax_h3_text_encoder_blocks_to_swap", "50"))
    add_arg(cmd, "--text_encoder_attn_mode", settings.get("minimax_h3_text_encoder_attn_mode") or "sdpa")
    target = str(settings.get("minimax_h3_training_target") or "").lower()
    legacy_video_only = bool(settings.get("minimax_h3_video_only"))
    add_arg(cmd, "--video_only", target.startswith("video only") or (not target and legacy_video_only))
    add_arg(cmd, "--audio_only", target.startswith("audio only"))
    add_arg(cmd, "--audio_loss_weight", settings.get("minimax_h3_audio_loss_weight") or "1.0")
    add_arg(cmd, "--h3_shift_video", settings.get("minimax_h3_shift_video") or "12.0")
    add_arg(cmd, "--h3_shift_audio", settings.get("minimax_h3_shift_audio") or "3.0")
    add_arg(cmd, "--h3_visual_cond_clean", settings.get("minimax_h3_visual_cond_clean") or "0.999")
    add_arg(cmd, "--h3_audio_cond_clean", settings.get("minimax_h3_audio_cond_clean") or "1.0")
    add_arg(cmd, "--h3_allow_experimental_sample_duration", settings.get("minimax_h3_allow_experimental_duration"))
    add_arg(cmd, "--network_module", "networks.lora_minimax_h3")
    add_arg(cmd, "--network_dim", settings.get("network_dim_low"))
    add_arg(cmd, "--network_alpha", settings.get("network_alpha_low"))
    if settings.get("minimax_h3_foundation_lora_enabled"):
        add_arg(cmd, "--h3_foundation_lora", settings.get("minimax_h3_foundation_lora"), is_path=True)
        add_arg(
            cmd,
            "--h3_foundation_lora_multiplier",
            settings.get("minimax_h3_foundation_lora_multiplier") or "1.0",
        )
    add_arg(cmd, "--blocks_to_swap", settings.get("blocks_to_swap"))
    add_automatic_swap_args(cmd, settings, add_arg)
    add_arg(cmd, "--convrot_int8_bwd", settings.get("minimax_h3_convrot_bwd_mode") or "bf16")
    teacher_matching = bool(settings.get("minimax_h3_teacher_matching"))
    protection = quality_protection_components(settings)
    if teacher_matching:
        add_arg(cmd, "--h3_teacher_matching", True)
        add_arg(cmd, "--h3_teacher_conditions", teacher_condition_value(settings.get("minimax_h3_teacher_conditions")))
        add_arg(cmd, "--h3_teacher_condition_sigma_max", settings.get("minimax_h3_teacher_condition_sigma_max") or "0.75")
        add_arg(cmd, "--h3_teacher_loss_dc_weight", settings.get("minimax_h3_teacher_loss_dc_weight") or "0.3")
        add_arg(cmd, "--h3_teacher_loss_mag_weight", settings.get("minimax_h3_teacher_loss_mag_weight") or "1.0")
        add_arg(cmd, "--h3_teacher_preservation_weight", settings.get("minimax_h3_teacher_preservation_weight") or "1.0")
        add_arg(cmd, "--h3_teacher_direct_loss_weight", settings.get("minimax_h3_teacher_direct_loss_weight") or "0.0")
        add_arg(cmd, "--h3_timestep_focus_min", settings.get("minimax_h3_timestep_focus_min") or "0.4")
        add_arg(cmd, "--h3_timestep_focus_max", settings.get("minimax_h3_timestep_focus_max") or "0.8")
        add_arg(cmd, "--h3_timestep_focus_prob", settings.get("minimax_h3_timestep_focus_prob") or "0.5")
    else:
        add_arg(cmd, "--h3_training_assistant_enabled", protection["assistant"])
        if protection["assistant"]:
            add_arg(
                cmd,
                "--h3_training_assistant",
                settings.get("minimax_h3_training_assistant") or DEFAULT_H3_TRAINING_ASSISTANT,
            )
    if not teacher_matching and protection["dynamic"]:
        add_arg(cmd, "--h3_guidance_loss_scale", settings.get("minimax_h3_guidance_distillation_scale") or "4.0")
        add_arg(cmd, "--h3_guidance_loss_sigma_min", settings.get("minimax_h3_guidance_distillation_sigma_min") or "0.15")
        add_arg(cmd, "--h3_guidance_loss_uncond_cache", _guidance_cache_path(settings), is_path=True)
    build_attention_arg(cmd, settings)
    build_sample_args(cmd, settings)
    # Official H3 draws one uniform base time and derives the linked video/audio sigmas
    # from it. Saved image-workflow recipes may carry Krea-style values in the shared
    # fields, so make a native-only view instead of emitting contradictory duplicate args.
    native_settings = dict(settings)
    native_settings.update(
        timestep_sampling="uniform",
        weighting_scheme="none",
        discrete_flow_shift="1.0",
    )
    build_common_train_args(cmd, native_settings)
    add_arg(cmd, "--weighting_scheme", "none")
    output_dir, output_name = build_output_dir(settings)
    add_arg(cmd, "--output_dir", output_dir, is_path=True)
    add_arg(cmd, "--output_name", output_name)
    return [cmd]


def build_cache_commands(settings, python_executable):
    if is_multimodal(settings):
        return _build_multimodal_cache_commands(settings, python_executable)
    commands = []
    if settings.get("recache_latents"):
        latent_command = [
                python_executable,
                "src/musubi_tuner/minimax_h3_image_cache_latents.py",
                "--dataset_config",
                settings["dataset_config"],
                "--vae",
                settings["vae_model"],
                "--vae_dtype",
                "float32",
            ]
        if settings.get("minimax_h3_teacher_matching") and teacher_condition_value(
            settings.get("minimax_h3_teacher_conditions")
        ) == "subject_ref":
            add_arg(latent_command, "--teacher_conditions", "subject_ref")
        commands.append(latent_command)
    if settings.get("recache_text"):
        command = [
            python_executable,
            "src/musubi_tuner/minimax_h3_image_cache_text_encoder_outputs.py",
            "--dataset_config",
            settings["dataset_config"],
            "--text_encoder",
            settings["minimax_h3_text_encoder"],
        ]
        tokenizer = settings.get("minimax_h3_tokenizer")
        if tokenizer:
            command.extend(["--tokenizer", tokenizer])
        add_arg(command, "--text_encoder_load_mode", settings.get("minimax_h3_text_encoder_load_mode") or "auto")
        add_arg(command, "--text_encoder_blocks_to_swap", setting_or_default(settings, "minimax_h3_text_encoder_blocks_to_swap", "50"))
        add_arg(command, "--cache_dtype", settings.get("minimax_h3_text_cache_dtype") or "bfloat16")
        add_arg(command, "--cache_h3_unconditional", quality_protection_components(settings)["dynamic"])
        if settings.get("minimax_h3_teacher_matching"):
            add_arg(command, "--teacher_conditions", teacher_condition_value(settings.get("minimax_h3_teacher_conditions")))
        build_dop_cache_args(command, settings)
        commands.append(command)
    return commands


def _build_multimodal_cache_commands(settings, python_executable):
    commands = []
    task = settings.get("minimax_h3_multimodal_task") or "t2va"
    teacher_matching = bool(settings.get("minimax_h3_teacher_matching"))
    if is_mixed_one_frame(settings) and (task == "ref2va" or teacher_matching):
        raise ValueError("Mixed native MiniMax image/video caching supports T2VA or FL2VA with teacher matching disabled")
    teacher_conditions = teacher_condition_value(settings.get("minimax_h3_teacher_conditions"))
    if teacher_matching and teacher_conditions == "first,last":
        latent_task = "fl2va"
    elif teacher_matching and teacher_conditions == "subject_ref":
        latent_task = "ref2va"
    else:
        latent_task = task
    if settings.get("recache_latents"):
        commands.append([
            python_executable, "src/musubi_tuner/minimax_h3_native_cache_latents.py",
            "--dataset_config", settings["dataset_config"],
            "--video_vae", settings["minimax_h3_video_vae"],
            "--audio_vae", settings["minimax_h3_audio_vae"],
            "--task", latent_task,
        ])
        add_arg(commands[-1], "--allow_experimental_duration", settings.get("minimax_h3_allow_experimental_duration"))
        add_arg(commands[-1], "--one_frame", is_mixed_one_frame(settings))
    if settings.get("recache_text"):
        command = [
            python_executable, "src/musubi_tuner/minimax_h3_native_cache_text_encoder_outputs.py",
            "--dataset_config", settings["dataset_config"],
            "--text_encoder", settings["minimax_h3_text_encoder"],
            "--task", task,
        ]
        add_arg(command, "--text_encoder_blocks_to_swap", setting_or_default(settings, "minimax_h3_text_encoder_blocks_to_swap", "50"))
        add_arg(command, "--text_encoder_attn_mode", settings.get("minimax_h3_text_encoder_attn_mode") or "sdpa")
        add_arg(command, "--text_cache_dtype", "bf16" if settings.get("minimax_h3_text_cache_dtype") == "bfloat16" else "float32")
        add_arg(command, "--one_frame", is_mixed_one_frame(settings))
        if teacher_matching:
            add_arg(command, "--teacher_conditions", teacher_conditions)
        elif quality_protection_components(settings)["dynamic"]:
            add_arg(command, "--uncond_output", _guidance_cache_path(settings), is_path=True)
        commands.append(command)
    return commands
