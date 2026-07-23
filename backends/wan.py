from backends._common import (
    add_arg, build_output_dir, build_network_args,
    build_attention_arg, build_common_train_args, build_sample_args,
)


def build_commands(settings):
    """Returns 1-2 accelerate launch commands for Wan 2.2 training."""
    commands = []
    train_low = settings.get("train_low_noise")
    train_high = settings.get("train_high_noise")

    dim_high = (settings.get("network_dim_high") or "").strip()
    alpha_high = (settings.get("network_alpha_high") or "").strip()
    is_separate_run = train_low and train_high and (dim_high or alpha_high)
    is_combined_run = train_low and train_high and not is_separate_run

    def _single(is_high_noise_run, is_combined_run):
        cmd = ["accelerate", "launch", "--num_processes", "1", "--num_cpu_threads_per_process", "1",
               "src/musubi_tuner/wan_train_network.py"]

        task_type = "i2v-A14B" if settings.get("is_i2v") else "t2v-A14B"
        add_arg(cmd, "--task", task_type)
        add_arg(cmd, "--mixed_precision", settings.get("mixed_precision"))
        add_arg(cmd, "--vae", settings.get("vae_model"), is_path=True)
        add_arg(cmd, "--t5", settings.get("t5_model"), is_path=True)
        add_arg(cmd, "--clip", settings.get("clip_model"), is_path=True)
        add_arg(cmd, "--dataset_config", settings.get("dataset_config"), is_path=True)

        if is_combined_run:
            add_arg(cmd, "--dit", settings.get("dit_low_noise"), is_path=True)
            add_arg(cmd, "--dit_high_noise", settings.get("dit_high_noise"), is_path=True)
            add_arg(cmd, "--timestep_boundary", settings.get("timestep_boundary"))
        else:
            dit = settings.get("dit_high_noise") if is_high_noise_run else settings.get("dit_low_noise")
            add_arg(cmd, "--dit", dit, is_path=True)
            if is_high_noise_run:
                add_arg(cmd, "--min_timestep", settings.get("min_timestep_high"))
                add_arg(cmd, "--max_timestep", settings.get("max_timestep_high"))
            else:
                add_arg(cmd, "--min_timestep", settings.get("min_timestep_low"))
                add_arg(cmd, "--max_timestep", settings.get("max_timestep_low"))

        # Network — use high-noise dim/alpha if this is a separate high-noise run
        if is_high_noise_run and not is_combined_run:
            dim = settings.get("network_dim_high") or settings.get("network_dim_low")
            alpha = settings.get("network_alpha_high") or settings.get("network_alpha_low")
            net_map = {"LoHa": "networks.loha", "LoKr": "networks.lokr"}
            net_type = settings.get("network_type", "LoRA")
            module = net_map.get(net_type, "networks.lora_wan")
            add_arg(cmd, "--network_module", module)
            add_arg(cmd, "--network_dim", dim)
            add_arg(cmd, "--network_alpha", alpha)
            if net_type == "LoKr":
                factor = (settings.get("lokr_factor") or "").strip()
                if factor and factor != "-1":
                    add_arg(cmd, "--network_args", f"factor={factor}")
        else:
            build_network_args(cmd, settings, "networks.lora_wan")

        build_attention_arg(cmd, settings)

        add_arg(cmd, "--fp8_base", settings.get("fp8_base"))
        add_arg(cmd, "--fp8_scaled", settings.get("fp8_scaled"))
        add_arg(cmd, "--fp8_t5", settings.get("fp8_t5"))
        add_arg(cmd, "--fp8_llm", settings.get("fp8_llm"))
        add_arg(cmd, "--force_v2_1_time_embedding", settings.get("force_v2_1_time_embedding"))

        if is_combined_run and settings.get("offload_inactive_dit"):
            add_arg(cmd, "--offload_inactive_dit", True)
        else:
            add_arg(cmd, "--blocks_to_swap", settings.get("blocks_to_swap"))

        build_sample_args(cmd, settings)
        build_common_train_args(cmd, settings)

        suffix = ""
        if train_low and train_high and not is_combined_run:
            suffix = "_HighNoise" if is_high_noise_run else "_LowNoise"
        elif train_high:
            suffix = "_HighNoise"
        elif train_low:
            suffix = "_LowNoise"

        output_dir, output_name = build_output_dir(settings, suffix)
        add_arg(cmd, "--output_dir", output_dir, is_path=True)
        add_arg(cmd, "--output_name", output_name)
        return cmd

    if is_separate_run:
        commands.append(_single(is_high_noise_run=False, is_combined_run=False))
        commands.append(_single(is_high_noise_run=True, is_combined_run=False))
    elif is_combined_run:
        commands.append(_single(is_high_noise_run=True, is_combined_run=True))
    elif train_low:
        commands.append(_single(is_high_noise_run=False, is_combined_run=False))
    elif train_high:
        commands.append(_single(is_high_noise_run=True, is_combined_run=False))
    return commands


def build_cache_commands(settings, python_executable, temp_config_fn=None):
    """Returns list of caching commands to prepend to the training sequence.

    temp_config_fn: optional callable(original_path) -> temp_path, strips
                    image_directory lines for I2V latent/text caching.
    """
    cmds = []
    is_i2v = settings.get("is_i2v")

    if settings.get("recache_latents"):
        cfg = settings["dataset_config"]
        if is_i2v and temp_config_fn:
            cfg = temp_config_fn(cfg)
        cmd = [python_executable, "src/musubi_tuner/wan_cache_latents.py",
               "--dataset_config", cfg, "--vae", settings["vae_model"]]
        if is_i2v:
            cmd.append("--i2v")
        cmds.append(cmd)

    if settings.get("recache_text"):
        cfg = settings["dataset_config"]
        if is_i2v and temp_config_fn:
            cfg = temp_config_fn(cfg)
        cmd = [python_executable, "src/musubi_tuner/wan_cache_text_encoder_outputs.py",
               "--dataset_config", cfg, "--t5", settings["t5_model"]]
        cmds.append(cmd)

    return cmds
