"""Shared preview identity and starter prompts for both interfaces."""
import os
from pathlib import Path


def standalone_preview_root():
    from prompt_library import default_library_root
    return default_library_root() / "preview_outputs"


def with_preview_output(settings):
    result = dict(settings)
    if not str(result.get("output_dir") or "").strip():
        result["output_dir"] = str(standalone_preview_root())
    return result


def preview_model_identity(settings):
    mode = str(settings.get("training_mode") or "")
    key = {"Krea 2": "krea2_dit_model", "MiniMax H3 (Experimental)": "minimax_h3_dit_model",
           "MOSS-Image (Experimental)": "moss_dit_model"}.get(mode, "dit_model")
    path = str(settings.get("krea2_turbo_dit") or "") if mode == "Krea 2" else ""
    path = path or str(settings.get(key) or "")
    return {"mode": mode, "model_key": os.path.normcase(os.path.abspath(path)) if path else "",
            "model_name": Path(path).name if path else mode}


def preview_metadata(settings, network_weights=""):
    return {**preview_model_identity(settings), "network_weights": network_weights,
            "baseline": not bool(network_weights),
            "sampling_lora": settings.get("moss_sample_lora", "") if str(settings.get("training_mode", "")).startswith("MOSS") else settings.get("krea2_turbo_lora", ""),
            "sampling_lora_strength": settings.get("moss_sample_lora_multiplier", "0.6") if str(settings.get("training_mode", "")).startswith("MOSS") else settings.get("krea2_turbo_lora_multiplier", "1.0")}


def prompt_defaults(settings):
    mode = settings.get("training_mode")
    common = dict(enabled=True, prompt="", neg="", seed=42)
    if mode == "MOSS-Image (Experimental)":
        return dict(common, width=1024, height=1024, steps=25, guidance=3, flow_shift=5, sampler="ddim", scheduler="simple")
    if mode == "MiniMax H3 (Experimental)":
        return dict(common, width=768, height=768, frames=39, fps=24, steps=20, guidance=1, cfg_scale=1, flow_shift=12)
    turbo = mode == "Krea 2" and (settings.get("krea2_turbo_dit") or settings.get("krea2_turbo_lora"))
    return dict(common, width=1024, height=1024, steps=8 if turbo else 28, guidance=1 if turbo else 5.5)


def starter_prompts():
    # Ordinary subjects the unmodified model can render: useful baseline tests
    # without a person's trigger token or a trained adapter.
    return [
        {"prompt": text, "seed": 42, "enabled": True, "_library_name": name}
        for name, text in [
            ("Portrait · skin and expression", "Close-up portrait of an adult woman with freckles and short curly hair, a gentle smile, soft window light, natural skin texture, eyes in sharp focus, photographic realism."),
            ("Full body · proportions", "Full-body photograph of an adult man wearing a plain blue shirt and beige trousers, standing beside a wooden chair, both hands visible and relaxed, natural proportions, soft daylight, clear detailed photography."),
            ("Hands · small details", "Close-up photograph of two adult hands carefully shaping a small clay bowl on a pottery table, realistic fingers and fingernails, damp clay texture, soft side lighting, sharp detail."),
            ("Object · texture and lighting", "A handmade red ceramic teapot on a wooden table beside a folded linen cloth, morning sunlight through a window, subtle glaze reflections, detailed still-life product photography."),
            ("Scene · composition", "A small bookshop with tall shelves, warm lamps and a green armchair beside a rainy window, wide composition, detailed wood and paper textures, realistic interior photograph."),
            ("Character · movement", "An adult woman in a yellow raincoat walks slowly along a quiet park path, turns her head and smiles, a light breeze moves her hair, overcast daylight, natural realistic appearance, medium-wide view."),
        ]
    ]
