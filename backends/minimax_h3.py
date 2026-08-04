"""Command construction for experimental MiniMax-H3 still-image LoRA training."""

from backends._common import (
    add_arg,
    build_attention_arg,
    build_common_train_args,
    build_output_dir,
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
    add_arg(cmd, "--dataset_config", settings.get("dataset_config"), is_path=True)
    add_arg(cmd, "--network_module", "networks.lora_minimax_h3")
    add_arg(cmd, "--network_dim", settings.get("network_dim_low"))
    add_arg(cmd, "--network_alpha", settings.get("network_alpha_low"))
    add_arg(cmd, "--blocks_to_swap", settings.get("blocks_to_swap"))
    add_arg(cmd, "--block_swap_h2d_only", True)
    add_arg(cmd, "--block_swap_ring_size", settings.get("block_swap_ring_size"))
    add_arg(cmd, "--use_pinned_memory_for_block_swap", settings.get("use_pinned_memory_for_block_swap"))
    add_arg(cmd, "--convrot_bwd_mode", settings.get("minimax_h3_convrot_bwd_mode") or "bf16")
    build_attention_arg(cmd, settings)
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
        commands.append(command)
    return commands
