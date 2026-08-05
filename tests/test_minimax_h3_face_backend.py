from pathlib import Path

from backends.minimax_h3_face import build_command


def test_h3_face_backend_forwards_quality_preview_settings(tmp_path: Path):
    settings = {
        "python_executable": "python",
        "minimax_h3_dit_model": str(tmp_path / "dit.safetensors"),
        "vae_model": str(tmp_path / "vae.safetensors"),
        "minimax_h3_text_encoder": str(tmp_path / "te.safetensors"),
        "attention_mechanism": "sdpa",
    }
    config = {
        "reference_dir": str(tmp_path / "refs"),
        "face_model_dir": str(tmp_path / "face-models"),
        "quality_preview_mode": "five_frame",
        "quality_preview_steps": 16,
        "quality_preview_final": False,
    }

    command = build_command(
        settings,
        config,
        tmp_path / "input.safetensors",
        tmp_path / "output.safetensors",
        tmp_path / "prompts.json",
    )

    assert command[command.index("--quality_preview_mode") + 1] == "five_frame"
    assert command[command.index("--quality_preview_steps") + 1] == "16"
    assert "--no-quality_preview_final" in command
