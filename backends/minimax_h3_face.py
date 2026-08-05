"""Command construction for experimental MiniMax-H3 DRaFT face refinement."""

from pathlib import Path

from backends._common import add_arg


def build_command(settings, config, input_lora, output_lora, prompts_json):
    command = [settings.get("python_executable") or "python", "src/musubi_tuner/minimax_h3_face_refinement.py"]
    add_arg(command, "--dit", settings.get("minimax_h3_dit_model"), is_path=True)
    add_arg(command, "--vae", settings.get("vae_model"), is_path=True)
    add_arg(command, "--text_encoder", settings.get("minimax_h3_text_encoder"), is_path=True)
    add_arg(command, "--tokenizer", settings.get("minimax_h3_tokenizer"))
    add_arg(command, "--text_encoder_load_mode", settings.get("minimax_h3_text_encoder_load_mode") or "auto")
    add_arg(command, "--network_weights", str(input_lora), is_path=True)
    add_arg(command, "--reference_dir", config.get("reference_dir"), is_path=True)
    add_arg(command, "--reference_manifest", config.get("reference_manifest"), is_path=True)
    add_arg(command, "--face_model_dir", config.get("face_model_dir"), is_path=True)
    add_arg(command, "--prompts_json", str(prompts_json), is_path=True)
    add_arg(command, "--output", str(output_lora), is_path=True)
    mapping = {
        "train_steps": "steps",
        "resolution": "resolution",
        "denoise_steps": "denoise_steps",
        "draft_k": "draft_k",
        "learning_rate": "learning_rate",
        "target_similarity": "target_similarity",
        "stop_similarity": "stop_similarity",
        "early_stop_patience": "early_stop_patience",
        "min_detection_rate": "min_detection_rate",
        "anti_copy_weight": "anti_copy_weight",
        "preview_every": "preview_every",
        "quality_preview_mode": "quality_preview_mode",
        "quality_preview_steps": "quality_preview_steps",
        "save_every": "save_every",
        "blocks_to_swap": "blocks_to_swap",
        "pose_reward_weight": "pose_reward_weight",
        "pose_min_references": "pose_min_references",
    }
    for argument, key in mapping.items():
        add_arg(command, f"--{argument}", config.get(key))
    add_arg(command, "--seed", config.get("seed", settings.get("seed")))
    add_arg(command, "--block_swap_ring_size", settings.get("block_swap_ring_size") or "2")
    add_arg(command, "--use_pinned_memory_for_block_swap", settings.get("use_pinned_memory_for_block_swap"))
    add_arg(command, "--convrot_bwd_mode", settings.get("minimax_h3_convrot_bwd_mode") or "bf16")
    if not config.get("qkvo_only", True):
        command.append("--no-attention_only")
    if not config.get("checkpoint_vae", True):
        command.append("--no-checkpoint_vae")
    if not config.get("quality_preview_final", True):
        command.append("--no-quality_preview_final")
    if config.get("pose_aware", False):
        command.append("--pose_aware")
    attention = settings.get("attention_mechanism", "sdpa")
    add_arg(command, "--attn_mode", {"sdpa": "torch", "none": "torch"}.get(attention, attention))
    if settings.get("split_attn"):
        command.append("--split_attn")
    return command


def output_path(settings, stage_label):
    name = f"{settings['output_name']}-{stage_label}"
    return Path(settings["output_dir"]) / name / f"{name}.safetensors"
