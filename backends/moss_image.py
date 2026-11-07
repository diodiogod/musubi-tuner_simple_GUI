"""Shared classic/modern GUI command construction for MOSS-Image."""
from backends._common import (
    add_arg, apply_cache_policy, build_output_dir, build_network_args,
    build_attention_arg, build_common_train_args, build_sample_args,
    build_dop_cache_args, build_dop_train_args,
)


def _text_options(cmd, settings):
    add_arg(cmd, "--text_encoder", settings.get("moss_text_encoder"), is_path=True)
    add_arg(cmd, "--tokenizer_cache_dir", settings.get("moss_tokenizer_cache_dir"), is_path=True)
    add_arg(cmd, "--moss_max_token_length", settings.get("moss_max_token_length", 4096))
    add_arg(cmd, "--text_encoder_cpu_streaming", settings.get("moss_text_encoder_cpu_streaming", True))


def build_commands(settings):
    cmd = ["accelerate", "launch", "--num_processes", "1", "--num_cpu_threads_per_process", "1",
           "src/musubi_tuner/moss_image_train_network.py"]
    add_arg(cmd, "--mixed_precision", settings.get("mixed_precision", "bf16"))
    add_arg(cmd, "--dit", settings.get("moss_dit_model"), is_path=True)
    add_arg(cmd, "--vae", settings.get("vae_model"), is_path=True)
    add_arg(cmd, "--dataset_config", settings.get("dataset_config"), is_path=True)
    _text_options(cmd, settings)
    add_arg(cmd, "--convrot_int8", settings.get("moss_convrot_int8", False))
    add_arg(cmd, "--convrot_bwd_mode", settings.get("moss_convrot_bwd_mode", "bf16"))
    add_arg(cmd, "--blocks_to_swap", settings.get("blocks_to_swap"))
    for name in ("weight_noise_sigma", "weight_noise_mode", "weight_noise_bound_norm",
                 "depth_anchor_weight", "depth_anchor_model", "depth_anchor_input_size",
                 "depth_anchor_gradient_weight", "depth_anchor_vae_device", "keep_depth_helpers_on_gpu"):
        add_arg(cmd, "--" + name, settings.get("moss_" + name))
    if not settings.get("moss_depth_anchor_grad_checkpoint", True):
        cmd.append("--no-depth_anchor_grad_checkpoint")
    if settings.get("moss_repa_enabled", False):
        from backends.moss_settings import DEFAULTS
        cmd.append("--moss_repa_enabled")
        for key in DEFAULTS:
            if key.startswith("moss_repa_") and key != "moss_repa_enabled":
                add_arg(cmd, "--" + key, settings.get(key, DEFAULTS[key]))
    add_arg(cmd, "--moss_sample_flow_shift", settings.get("moss_sample_flow_shift", 5.0))
    for key in ("moss_sample_sampler", "moss_sample_scheduler", "moss_sample_lora", "moss_sample_lora_multiplier"):
        add_arg(cmd, "--" + key, settings.get(key), is_path=key == "moss_sample_lora")
    if settings.get("moss_pretraining_schedule", True):
        cmd.append("--moss_pretraining_schedule")
        for key in ("moss_logit_mean", "moss_logit_std", "moss_timestep_cutoff", "moss_timestep_uniform_ratio"):
            add_arg(cmd, "--" + key, settings.get(key))
    # Preserve an explicit empty value: it disables the official skip-layer guidance.
    cmd.extend(["--moss_sample_skip_layers", str(settings.get("moss_sample_skip_layers", "22"))])
    target_args = []
    if settings.get("moss_lora_target_preset") == "Attention only":
        target_args = ["exclude_patterns=['.*[.]ffn_up_gate','.*[.]ffn_down']"]
    build_network_args(cmd, settings, "networks.lora_moss_image", target_args)
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
        commands.append([python_executable, "src/musubi_tuner/moss_image_cache_latents.py",
                         "--dataset_config", settings["dataset_config"], "--vae", settings["vae_model"]])
    if settings.get("recache_text"):
        cmd = [python_executable, "src/musubi_tuner/moss_image_cache_text_encoder_outputs.py",
               "--dataset_config", settings["dataset_config"]]
        _text_options(cmd, settings)
        build_dop_cache_args(cmd, settings)
        commands.append(cmd)
    return apply_cache_policy(commands, settings)
