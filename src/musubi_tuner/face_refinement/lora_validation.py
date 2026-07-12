"""Lightweight validation for Krea 2 LoRAs used by face refinement."""

from pathlib import Path

from safetensors import safe_open


def validate_krea2_lora(path: str | Path) -> dict:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise ValueError("Choose an existing LoRA .safetensors file.")
    if candidate.suffix.lower() != ".safetensors":
        raise ValueError("The refinement input must be a .safetensors LoRA.")

    try:
        with safe_open(candidate, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        raise ValueError(f"Could not read the selected LoRA: {exc}") from exc

    down = [key for key in keys if key.startswith("lora_unet_blocks_") and key.endswith(".lora_down.weight")]
    up = [key for key in keys if key.startswith("lora_unet_blocks_") and key.endswith(".lora_up.weight")]
    krea_attention = [key for key in down if any(part in key for part in ("_attn_wq.", "_attn_wk.", "_attn_wv.", "_attn_wo."))]
    if not down or len(down) != len(up) or not krea_attention:
        raise ValueError(
            "This file does not look like a complete Musubi Krea 2 LoRA. "
            "Select LoRA weights, not a base model, optimizer file, or LoRA for another architecture."
        )
    return {"path": str(candidate.resolve()), "modules": len(down), "attention_modules": len(krea_attention)}


def render_trigger_prompts(prompts: list[str], trigger_word: str) -> list[str]:
    from musubi_tuner.face_refinement.pose import POSE_BUCKETS, TAG_PATTERN

    trigger = trigger_word.strip()
    rendered = []
    for prompt in prompts:
        text = str(prompt).strip()
        if not text:
            continue
        tag_prefix = ""
        tag_match = TAG_PATTERN.match(text)
        if tag_match and tag_match.group(1).lower() in ("auto", *POSE_BUCKETS):
            tag_prefix = f"[{tag_match.group(1).lower()}] "
            text = text[tag_match.end():].strip()
        if "{trigger}" in text:
            text = text.replace("{trigger}", trigger or "the person")
        elif trigger and trigger.casefold() not in text.casefold():
            text = f"{trigger}, {text}"
        rendered.append(tag_prefix + text)
    return rendered
