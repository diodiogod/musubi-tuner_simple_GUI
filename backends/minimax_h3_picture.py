"""GUI wiring for the dedicated MiniMax H3 Image checkpoint."""

from pathlib import Path

from backends import minimax_h3


MODE = "MiniMax H3 Image (Experimental)"
MODEL_FIELDS = {
    "minimax_h3_picture_dit": (
        "H3 Image DiT",
        "Choose minimax_h3_image_pruned_int8_convrot.safetensors. This checkpoint has six image output heads.",
    ),
    "minimax_h3_picture_vae": (
        "H3 Image VAE",
        "Choose minimax_h3_image_vae_fp16.safetensors. This VAE decodes a single image directly.",
    ),
    "minimax_h3_picture_text_encoder": (
        "Qwen3-VL Text Encoder",
        "Choose the shared Qwen3-VL-32B MiniMax H3 text encoder in ConvRot INT8 or NVFP4 AWQ.",
    ),
    "minimax_h3_picture_assistant": (
        "H3 Image Training Adapter (Optional)",
        "Choose a frozen training adapter specifically for MiniMax H3 Image. It can help preserve the model’s look. It is switched off for previews and stays separate from your saved LoRA.",
    ),
}
WEIGHTS_URL = "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main"
ASSISTANT_URL = "https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter"


def runtime_settings(settings):
    prepared = {key: value for key, value in settings.items() if not key.startswith(("minimax_h3_", "krea2_", "dop_"))}
    assistant = str(settings.get("minimax_h3_picture_assistant") or "").strip()
    if assistant and not settings.get("_preview_only") and not Path(assistant).expanduser().is_file():
        raise ValueError("Choose a local H3 Image training adapter .safetensors file.")
    prepared.update(
        minimax_h3_dit_model=settings.get("minimax_h3_picture_dit") or "",
        vae_model=settings.get("minimax_h3_picture_vae") or "",
        minimax_h3_text_encoder=settings.get("minimax_h3_picture_text_encoder") or "",
        minimax_h3_training_assistant=assistant,
        minimax_h3_training_assistant_enabled=bool(assistant),
        minimax_h3_dynamic_sigma_enabled=False,
        minimax_h3_base_preservation_enabled=False,
        minimax_h3_text_encoder_blocks_to_swap=settings.get("minimax_h3_text_encoder_blocks_to_swap") or "50",
        minimax_h3_tokenizer=settings.get("minimax_h3_tokenizer") or "MiniMaxAI/MiniMax-H3",
        mixed_precision="bf16", compile=False, fp8_base=False, fp8_scaled=False,
        timestep_sampling="uniform", gradient_checkpointing=True, num_timestep_buckets=None,
    )
    return prepared


def build_cache_commands(settings, python_executable):
    return [command + ["--h3_picture_model"] for command in
            minimax_h3.build_cache_commands(runtime_settings(settings), python_executable)]


def build_commands(settings):
    return [command + ["--h3_picture_model"] for command in minimax_h3.build_commands(runtime_settings(settings))]
