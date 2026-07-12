"""Command construction for the isolated Krea 2 DRaFT face-refinement stage."""

from pathlib import Path

from backends._common import add_arg


def build_command(settings, config, input_lora, output_lora, prompts_json):
    command = [settings.get("python_executable") or "python", "src/musubi_tuner/krea2_face_refinement.py"]
    add_arg(command, "--dit", settings.get("krea2_dit_model"), is_path=True)
    add_arg(command, "--vae", settings.get("vae_model"), is_path=True)
    add_arg(command, "--text_encoder", settings.get("krea2_text_encoder"), is_path=True)
    add_arg(command, "--network_weights", str(input_lora), is_path=True)
    add_arg(command, "--reference_dir", config.get("reference_dir"), is_path=True)
    add_arg(command, "--face_model_dir", config.get("face_model_dir"), is_path=True)
    add_arg(command, "--prompts_json", str(prompts_json), is_path=True)
    add_arg(command, "--output", str(output_lora), is_path=True)
    mapping = {
        "train_steps": "steps", "resolution": "resolution", "denoise_steps": "denoise_steps",
        "draft_k": "draft_k", "cfg_scale": "cfg_scale", "learning_rate": "learning_rate",
        "target_similarity": "target_similarity", "stop_similarity": "stop_similarity",
        "early_stop_patience": "early_stop_patience", "min_detection_rate": "min_detection_rate",
        "anti_copy_weight": "anti_copy_weight", "preview_every": "preview_every", "save_every": "save_every",
        "blocks_to_swap": "blocks_to_swap",
        "gpu_id": "gpu_id",
    }
    for argument, key in mapping.items():
        add_arg(command, f"--{argument}", config.get(key))
    add_arg(command, "--seed", config.get("seed", settings.get("seed")))
    if not config.get("qkvo_only", True):
        command.append("--no-qkvo_only")
    if not config.get("checkpoint_vae", True):
        command.append("--no-checkpoint_vae")
    if settings.get("fp8_scaled"):
        command.append("--fp8_scaled")
    attention = settings.get("attention_mechanism", "sdpa")
    add_arg(command, "--attn_mode", {"sdpa": "torch", "none": "torch"}.get(attention, attention))
    if settings.get("split_attn"):
        command.append("--split_attn")
    add_arg(command, "--projector_diff", settings.get("krea2_projector_diff"), is_path=True)
    add_arg(command, "--projector_diff_strength", settings.get("krea2_projector_diff_strength"))
    return command


def output_path(settings, stage_label):
    name = f"{settings['output_name']}-{stage_label}"
    return Path(settings["output_dir"]) / name / f"{name}.safetensors"
