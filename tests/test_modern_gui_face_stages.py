import json
from pathlib import Path

from modern_gui.face_stages import prepare_face_stage


def test_face_stage_builds_prompt_and_reference_manifests(tmp_path: Path):
    input_lora = tmp_path / "input.safetensors"
    input_lora.write_bytes(b"lora")
    reference = tmp_path / "face.png"
    reference.write_bytes(b"image")
    base = {
        "training_mode": "Krea 2",
        "output_name": "portrait",
        "output_dir": str(tmp_path),
        "krea2_dit_model": "raw.safetensors",
        "krea2_text_encoder": "text",
        "vae_model": "vae.safetensors",
        "face_refinement_config": {
            "trigger_word": "sks",
            "prompts": ["studio portrait"],
            "reference_dir": str(tmp_path),
            "face_model_dir": str(tmp_path / "models"),
            "preflight_report": {
                "scored_images": [
                    {"path": str(reference), "bucket": "front", "confidence": 0.9}
                ]
            },
        },
    }

    settings, command, output = prepare_face_stage(
        base, {"label": "identity", "type": "face_refinement", "steps": "30"}, 0, input_lora
    )

    assert settings["stage_type"] == "face_refinement"
    assert settings["resume_exact_position"] is False
    assert settings["network_weights"] == str(input_lora)
    assert output == tmp_path / "portrait-identity" / "portrait-identity.safetensors"
    assert "--network_weights" in command
    prompts_path = Path(command[command.index("--prompts_json") + 1])
    manifest_path = Path(command[command.index("--reference_manifest") + 1])
    assert json.loads(prompts_path.read_text(encoding="utf-8"))["prompts"] == ["sks, studio portrait"]
    references = json.loads(manifest_path.read_text(encoding="utf-8"))["reference_images"]
    assert references[0]["enabled"] is True
