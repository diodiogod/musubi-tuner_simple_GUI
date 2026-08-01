import json

from modern_gui import settings


def test_schema_exposes_every_scalar_setting_and_keeps_structured_editors_separate():
    payload = {
        "training_mode": "Krea 2",
        "dataset_config": "dataset.toml",
        "learning_rate": "2e-4",
        "gradient_checkpointing": True,
        "krea2_depth_anchor_weight": "0.1",
        "sample_prompts_data": [{"prompt": "test"}],
        "staged_training_config": [{"label": "1024"}],
        "face_refinement_config": {"steps": 20},
    }

    schema = settings.settings_schema(payload)
    keys = {field["key"] for section in schema["sections"] for field in section["fields"]}

    assert keys == set(payload) - settings.STRUCTURED_KEYS
    assert schema["field_count"] == len(keys)
    assert set(schema["structured_keys"]) == settings.STRUCTURED_KEYS


def test_mode_specific_fields_are_tagged_for_frontend_filtering():
    schema = settings.settings_schema(
        {
            "dit_high_noise": "high.safetensors",
            "flux2_dit_model": "flux.safetensors",
            "krea2_depth_anchor_weight": "0",
            "learning_rate": "2e-4",
        }
    )
    fields = {field["key"]: field for section in schema["sections"] for field in section["fields"]}

    assert fields["dit_high_noise"]["modes"] == ["Wan 2.2"]
    assert fields["flux2_dit_model"]["modes"] == ["Flux.2 Klein", "Flux.2 Dev"]
    assert fields["krea2_depth_anchor_weight"]["modes"] == ["Krea 2"]
    assert "modes" not in fields["learning_rate"]


def test_atomic_settings_save_preserves_nested_specialized_state(monkeypatch, tmp_path):
    destination = tmp_path / "last_settings.json"
    monkeypatch.setattr(settings, "LAST_SETTINGS", destination)
    payload = {
        "sample_prompts_data": [{"prompt": "one", "enabled": True}],
        "staged_training_config": [{"type": "face_refinement", "enabled": True}],
        "face_refinement_config": {"pose_plan": {"enabled": True}},
    }

    settings.save_settings(payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert not destination.with_suffix(".json.tmp").exists()
