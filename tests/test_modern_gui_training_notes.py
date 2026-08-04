from pathlib import Path

from modern_gui.training_notes import effective_training_comment, training_settings_summary


def test_generated_summary_stays_separate_and_is_stage_aware(tmp_path: Path):
    dataset = tmp_path / "portraits.toml"
    dataset.write_text(
        "[[datasets]]\nresolution = [1024, 1024]\n",
        encoding="utf-8",
    )
    settings = {
        "training_mode": "Krea 2",
        "output_name": "identity",
        "network_type": "LoRA",
        "network_dim_low": "32",
        "network_alpha_low": "16",
        "dataset_config": str(dataset),
        "learning_rate": "2e-4",
        "optimizer_type": "AdamW8bit",
        "use_staged_training": True,
        "staged_training_config": [
            {"enabled": True, "label": "512", "epochs": "2", "type": "standard"},
            {"enabled": True, "label": "face", "steps": "30", "type": "face_refinement"},
        ],
        "training_comment": "Keep skin texture.",
        "auto_training_settings_summary": True,
    }

    summary = training_settings_summary(settings)
    comment = effective_training_comment(settings)

    assert "staged 512px 2 epochs → face 30 steps face" in summary
    assert comment.startswith("Keep skin texture.\n\nSettings:")


def test_generated_summary_can_be_disabled():
    settings = {
        "training_mode": "Wan 2.2",
        "training_comment": "custom only",
        "auto_training_settings_summary": False,
    }

    assert effective_training_comment(settings) == "custom only"


def test_stage_without_explicit_enabled_flag_is_included_in_summary():
    settings = {
        "training_mode": "Krea 2",
        "output_name": "portrait",
        "network_type": "LoRA",
        "use_staged_training": True,
        "staged_training_config": [{"label": "512", "epochs": "2"}],
    }

    assert "staged 512px 2 epochs" in training_settings_summary(settings)
