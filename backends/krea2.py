from backends._common import (
    add_arg, build_output_dir, build_network_args,
    build_attention_arg, build_common_train_args, build_sample_args,
    build_dop_cache_args, build_dop_train_args,
)


def build_commands(settings):
    """Returns a single accelerate launch command for Krea 2 training."""
    cmd = ["accelerate", "launch", "--num_processes", "1", "--num_cpu_threads_per_process", "1",
           "src/musubi_tuner/krea2_train_network.py"]

    add_arg(cmd, "--mixed_precision", settings.get("mixed_precision"))
    add_arg(cmd, "--dit", settings.get("krea2_dit_model"), is_path=True)
    add_arg(cmd, "--vae", settings.get("vae_model"), is_path=True)
    add_arg(cmd, "--dataset_config", settings.get("dataset_config"), is_path=True)
    add_arg(cmd, "--text_encoder", settings.get("krea2_text_encoder"), is_path=True)
    add_arg(cmd, "--turbo_dit", settings.get("krea2_turbo_dit"), is_path=True)
    add_arg(cmd, "--turbo_dit_cache", settings.get("krea2_turbo_dit_cache"))
    add_arg(cmd, "--projector_diff", settings.get("krea2_projector_diff"), is_path=True)
    add_arg(cmd, "--projector_diff_strength", settings.get("krea2_projector_diff_strength"))
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

    build_network_args(cmd, settings, "networks.lora_krea2")
    build_attention_arg(cmd, settings)

    add_arg(cmd, "--fp8_base", settings.get("fp8_base"))
    add_arg(cmd, "--fp8_scaled", settings.get("fp8_scaled"))
    add_arg(cmd, "--blocks_to_swap", settings.get("blocks_to_swap"))

    build_sample_args(cmd, settings)
    build_dop_train_args(cmd, settings)
    build_common_train_args(cmd, settings)

    output_dir, output_name = build_output_dir(settings)
    add_arg(cmd, "--output_dir", output_dir, is_path=True)
    add_arg(cmd, "--output_name", output_name)

    return [cmd]


def build_cache_commands(settings, python_executable):
    """Returns list of caching commands for Krea 2."""
    cmds = []

    if settings.get("recache_latents"):
        cmd = [python_executable, "src/musubi_tuner/krea2_cache_latents.py",
               "--dataset_config", settings["dataset_config"],
               "--vae", settings["vae_model"]]
        cmds.append(cmd)

    if settings.get("recache_text"):
        cmd = [python_executable, "src/musubi_tuner/krea2_cache_text_encoder_outputs.py",
               "--dataset_config", settings["dataset_config"],
               "--text_encoder", settings["krea2_text_encoder"]]
        build_dop_cache_args(cmd, settings)
        cmds.append(cmd)

    return cmds
