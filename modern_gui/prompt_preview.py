from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from modern_gui.sample_prompts import serialize_sample_prompt


def serialize_prompt(prompt: dict[str, Any]) -> str:
    return serialize_sample_prompt(prompt, "Krea 2")


def resolve_preview_lora(settings: dict[str, Any]) -> str:
    explicit_value = str(settings.get("network_weights") or "").strip()
    if explicit_value:
        explicit = Path(explicit_value).expanduser()
        if explicit.is_file():
            return str(explicit)
    output_root = Path(str(settings.get("output_dir") or "")).expanduser()
    output_name = str(settings.get("output_name") or "").strip()
    if not output_name or not output_root.is_dir():
        return ""
    exact = output_root / output_name
    roots = [exact] if exact.is_dir() else [path for path in output_root.glob(f"{output_name}*") if path.is_dir()]
    candidates = [
        path for root in roots for path in root.glob("*.safetensors")
        if path.is_file() and "optimizer" not in path.name.lower()
    ]
    return str(max(candidates, key=lambda path: path.stat().st_mtime)) if candidates else ""


def _write_batch_prompt_snapshot(save_path: Path, prompts: list[dict[str, Any]]) -> Path:
    """Write one immutable batch input so concurrent previews cannot overwrite it."""

    destination = save_path / f"preview_prompts_{uuid.uuid4().hex}.txt"
    payload = "\n".join(serialize_prompt(item) for item in prompts)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".preview-prompts-",
        suffix=".tmp",
        dir=save_path,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return destination


def build_krea_preview(settings: dict[str, Any], prompts: list[dict[str, Any]]) -> tuple[list[str], Path]:
    if settings.get("training_mode") != "Krea 2":
        raise ValueError("Standalone sample preview is currently supported for Krea 2 only.")
    required = {
        "DiT model": settings.get("krea2_dit_model"),
        "VAE model": settings.get("vae_model"),
        "Text encoder": settings.get("krea2_text_encoder"),
    }
    missing = [name for name, value in required.items() if not value or not Path(str(value)).is_file()]
    if missing:
        raise ValueError("Missing required Krea 2 paths: " + ", ".join(missing))
    enabled = [prompt for prompt in prompts if prompt.get("enabled", True)]
    if not enabled:
        raise ValueError("Enable at least one sample prompt.")
    turbo = str(settings.get("krea2_turbo_dit") or "").strip()
    dit = Path(turbo) if turbo else Path(str(settings["krea2_dit_model"]))
    if not dit.is_file():
        raise ValueError("The selected Krea 2 inference DiT does not exist.")
    save_path = Path(str(settings.get("output_dir") or "")).expanduser() / (
        str(settings.get("output_name") or "").strip() or "krea2_test"
    ) / "sample_test"
    save_path.mkdir(parents=True, exist_ok=True)
    attention = {"sdpa": "torch", "flash_attn": "flash", "sage_attn": "sageattn"}.get(
        settings.get("attention_mechanism"), settings.get("attention_mechanism") or "torch"
    )
    command = [
        sys.executable, "src/musubi_tuner/krea2_generate_image.py",
        "--dit", str(dit), "--vae", str(settings["vae_model"]),
        "--text_encoder", str(settings["krea2_text_encoder"]),
        "--save_path", str(save_path), "--attn_mode", str(attention),
    ]
    if turbo:
        command.append("--turbo")
    if len(enabled) == 1:
        prompt = enabled[0]
        command.insert(2, str(prompt.get("prompt", "")))
        mapping = {
            "negative_prompt": "neg", "width": "width", "height": "height",
            "steps": "steps", "guidance_scale": "guidance", "seed": "seed",
            "mu": "mu", "y1": "y1", "y2": "y2",
        }
        for flag, key in mapping.items():
            value = str(prompt.get(key, "")).strip()
            if value:
                command.extend([f"--{flag}", value])
    else:
        prompt_file = _write_batch_prompt_snapshot(save_path, enabled)
        command.extend(["--from_file", str(prompt_file)])
    if settings.get("fp8_scaled"):
        command.append("--fp8_scaled")
    if str(settings.get("blocks_to_swap") or "").strip() not in {"", "0"}:
        command.extend(["--blocks_to_swap", str(settings["blocks_to_swap"])])
    if settings.get("krea2_projector_diff"):
        command.extend(["--projector_diff", str(settings["krea2_projector_diff"])])
        if str(settings.get("krea2_projector_diff_strength") or "").strip():
            command.extend(["--projector_diff_strength", str(settings["krea2_projector_diff_strength"])])
    lora = resolve_preview_lora(settings)
    if lora:
        command.extend(["--lora_weight", lora])
    return command, save_path
