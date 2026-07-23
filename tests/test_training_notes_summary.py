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
