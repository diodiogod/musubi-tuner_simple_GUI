"""Command construction for experimental MiniMax-H3 still-image LoRA training."""

DEFAULT_H3_TRAINING_ASSISTANT = (
    "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_alpha.safetensors"
)


def quality_protection_method(settings):
    """Return the CLI value while accepting old saved boolean recipes."""
    value = str(settings.get("minimax_h3_quality_protection_method") or "").strip().lower()
    aliases = {
        "dynamic sigma (recommended)": "dynamic",
        "ostris assistant (alpha)": "assistant",
        "assistant + base preservation (alpha)": "assistant_preservation",
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
    add_arg(
        cmd,
        "--minimax_h3_preview_decode_min_free_gb",
        settings.get("minimax_h3_preview_decode_min_free_gb") or "9.0",
    )
    add_arg(cmd, "--network_module", "networks.lora_minimax_h3")
    add_arg(cmd, "--network_dim", settings.get("network_dim_low"))
    add_arg(cmd, "--network_alpha", settings.get("network_alpha_low"))
    add_arg(cmd, "--blocks_to_swap", settings.get("blocks_to_swap"))
    add_arg(cmd, "--block_swap_h2d_only", True)
    add_arg(cmd, "--block_swap_ring_size", settings.get("block_swap_ring_size"))
    add_arg(cmd, "--use_pinned_memory_for_block_swap", settings.get("use_pinned_memory_for_block_swap"))
    add_arg(cmd, "--convrot_bwd_mode", settings.get("minimax_h3_convrot_bwd_mode") or "bf16")
    protection = quality_protection_components(settings)
    add_arg(cmd, "--h3_training_assistant_enabled", protection["assistant"])
    add_arg(cmd, "--h3_guidance_distillation_protection", protection["dynamic"])
    add_arg(cmd, "--h3_dynamic_sigma_every_n_steps", settings.get("minimax_h3_dynamic_sigma_every_n_steps") or "1")
    add_arg(cmd, "--h3_guidance_distillation_scale", settings.get("minimax_h3_guidance_distillation_scale") or "4.0")
    add_arg(cmd, "--h3_guidance_distillation_schedule", settings.get("minimax_h3_guidance_distillation_schedule") or "sigma")
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


def build_cache_commands(settings, python_executable):
    commands = []
    if settings.get("recache_latents"):
        commands.append(
            [
                python_executable,
                "src/musubi_tuner/minimax_h3_image_cache_latents.py",
                "--dataset_config",
                settings["dataset_config"],
                "--vae",
                settings["vae_model"],
                "--vae_dtype",
                "float32",
            ]
        )
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
        add_arg(command, "--cache_dtype", settings.get("minimax_h3_text_cache_dtype") or "bfloat16")
        add_arg(command, "--cache_h3_unconditional", quality_protection_components(settings)["dynamic"])
        build_dop_cache_args(command, settings)
        commands.append(command)
    return commands
