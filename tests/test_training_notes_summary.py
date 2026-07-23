from musubi_tuner_gui import MusubiTunerGUI


def test_krea_summary_captures_identity_and_regularization_choices(tmp_path):
    dataset = tmp_path / "character.toml"
    dataset.write_text('[[datasets]]\nresolution = [1024, 768]\n', encoding="utf-8")
    summary = MusubiTunerGUI._training_settings_summary(
        {
            "training_mode": "Krea 2",
            "output_name": "alice-v2",
            "network_type": "LoRA",
            "network_dim_low": "16",
            "network_alpha_low": "16",
            "max_train_epochs": "3",
            "dataset_config": str(dataset),
            "learning_rate": "1e-4",
            "optimizer_type": "adamw8bit",
            "dop_enabled": True,
            "dop_loss_weight": "1.0",
            "dop_class_word": "a woman",
            "krea2_depth_anchor_weight": "0.01",
            "krea2_depth_anchor_input_size": "518",
            "krea2_keep_depth_helpers_on_gpu": True,
            "krea2_weight_noise_sigma": "0.0125",
            "krea2_weight_noise_mode": "relative",
            "krea2_projector_diff": "C:/models/filter-patch.safetensors",
            "krea2_projector_diff_strength": "1.8",
            "blocks_to_swap": "10",
        }
    )
    assert "run=alice-v2" in summary
    assert "LoRA rank 16 α16" in summary
    assert "data=character (1024×768)" in summary
    assert "DOP 1 (a woman)" in summary
    assert "depth 0.01@518 GPU" in summary
    assert "weight-noise 0.0125 relative" in summary
    assert "projector=filter-patch.safetensors@1.8" in summary


def test_staged_summary_uses_enabled_stage_labels_and_limits():
    summary = MusubiTunerGUI._training_settings_summary(
        {
            "training_mode": "Krea 2",
            "network_type": "LoRA",
            "network_dim_low": "8",
            "use_staged_training": True,
            "staged_training_config": [
                {"enabled": True, "label": "512", "type": "standard", "epochs": "1", "steps": ""},
                {"enabled": True, "label": "face", "type": "face_refinement", "epochs": "", "steps": "30"},
                {"enabled": False, "label": "disabled", "type": "standard", "epochs": "9", "steps": ""},
            ],
        }
    )
    assert "staged 512px 1 epochs → face 30 steps face" in summary
    assert "disabled" not in summary


def test_live_summary_is_separate_from_custom_note_and_rebuilt():
    base = {
        "training_mode": "Krea 2",
        "training_comment": "Compare skin texture with the previous run.",
        "auto_training_settings_summary": True,
        "network_type": "LoRA",
        "network_dim_low": "8",
        "max_train_epochs": "2",
    }
    first = MusubiTunerGUI._effective_training_comment(base)
    second = MusubiTunerGUI._effective_training_comment(base | {"network_dim_low": "16"})

    assert first.startswith("Compare skin texture with the previous run.\n\nSettings:")
    assert "LoRA rank 8" in first
    assert "LoRA rank 16" in second
    assert "LoRA rank 8" not in second


def test_disabled_live_summary_preserves_only_custom_note():
    settings = {
        "training_comment": "My own note",
        "auto_training_settings_summary": False,
        "training_mode": "Krea 2",
        "network_dim_low": "16",
    }
    assert MusubiTunerGUI._effective_training_comment(settings) == "My own note"


def test_standard_stage_summary_uses_that_stages_effective_limit():
    summary = MusubiTunerGUI._training_settings_summary(
        {
            "training_mode": "Krea 2",
            "output_name": "alice-1024px",
            "network_type": "LoRA",
            "network_dim_low": "16",
            "use_staged_training": True,
            "stage_type": "standard",
            "max_train_steps": "250",
            "max_train_epochs": "",
            "dataset_config": "C:/datasets/1024.toml",
            "staged_training_config": [
                {"enabled": True, "label": "512", "type": "standard", "epochs": "1", "steps": ""},
                {"enabled": True, "label": "1024", "type": "standard", "epochs": "", "steps": "250"},
            ],
        }
    )
    assert "250 steps" in summary
    assert "data=1024" in summary
    assert "staged " not in summary
