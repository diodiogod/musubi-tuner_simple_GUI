from backends._common import (
    add_arg, build_output_dir, build_network_args,
    build_attention_arg, build_common_train_args, build_sample_args,
    build_dop_cache_args, build_dop_train_args,
)

FLUX2_VERSION_MAP = {
    "Klein Base 4B ★": "klein-base-4b",
    "Klein 4B": "klein-4b",
    "Klein Base 9B ★": "klein-base-9b",
    "Klein 9B": "klein-9b",
    "Dev": "dev",
}


def _get_version(settings):
    return FLUX2_VERSION_MAP.get(
        settings.get("flux2_model_version", "Klein Base 4B ★"), "klein-base-4b"
    )


def build_commands(settings):
    """Returns a single accelerate launch command for Flux.2 training."""
    cmd = ["accelerate", "launch", "--num_processes", "1", "--num_cpu_threads_per_process", "1",
           "src/musubi_tuner/flux_2_train_network.py"]

    add_arg(cmd, "--model_version", _get_version(settings))
    add_arg(cmd, "--mixed_precision", settings.get("mixed_precision"))
    add_arg(cmd, "--dit", settings.get("flux2_dit_model"), is_path=True)
    add_arg(cmd, "--vae", settings.get("vae_model"), is_path=True)
    add_arg(cmd, "--text_encoder", settings.get("flux2_text_encoder"), is_path=True)
    add_arg(cmd, "--dataset_config", settings.get("dataset_config"), is_path=True)

    build_network_args(cmd, settings, "networks.lora_flux_2")
    build_attention_arg(cmd, settings)

    add_arg(cmd, "--fp8_base", settings.get("fp8_base"))
    add_arg(cmd, "--fp8_scaled", settings.get("fp8_scaled"))
    if settings.get("fp8_text_encoder"):
        cmd.append("--fp8_text_encoder")

    add_arg(cmd, "--blocks_to_swap", settings.get("blocks_to_swap"))

    build_sample_args(cmd, settings)
    if _get_version(settings) != "dev":
        build_dop_train_args(cmd, settings)
    build_common_train_args(cmd, settings)

    output_dir, output_name = build_output_dir(settings)
    add_arg(cmd, "--output_dir", output_dir, is_path=True)
    add_arg(cmd, "--output_name", output_name)

    return [cmd]


def build_cache_commands(settings, python_executable):
    """Returns list of caching commands for Flux.2."""
    cmds = []
    ver = _get_version(settings)

    if settings.get("recache_latents"):
        cmd = [python_executable, "src/musubi_tuner/flux_2_cache_latents.py",
               "--dataset_config", settings["dataset_config"],
               "--vae", settings["vae_model"],
               "--model_version", ver]
        cmds.append(cmd)

    if settings.get("recache_text"):
        cmd = [python_executable, "src/musubi_tuner/flux_2_cache_text_encoder_outputs.py",
               "--dataset_config", settings["dataset_config"],
               "--text_encoder", settings["flux2_text_encoder"],
               "--model_version", ver]
        if settings.get("fp8_text_encoder"):
            cmd.append("--fp8_text_encoder")
        if ver != "dev":
            build_dop_cache_args(cmd, settings)
        cmds.append(cmd)

    return cmds
