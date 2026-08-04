import json
from pathlib import Path

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
    assert fields["krea2_depth_anchor_weight"]["modes"] == ["Krea 2", settings.MINIMAX_H3_MODE]
    assert "modes" not in fields["learning_rate"]


def test_minimax_fields_and_fixed_controls_are_mode_aware():
    schema = settings.settings_schema(
        {
            "minimax_h3_dit_model": "h3.safetensors",
            "minimax_h3_convrot_bwd_mode": "bf16",
            "network_type": "LoRA",
            "compile": False,
        }
    )
    fields = {field["key"]: field for section in schema["sections"] for field in section["fields"]}

    assert fields["minimax_h3_dit_model"]["modes"] == [settings.MINIMAX_H3_MODE]
    assert fields["minimax_h3_dit_model"]["label"] == "Pruned ConvRot INT8 DiT (Required)"
    assert fields["minimax_h3_convrot_bwd_mode"]["options"] == ["bf16", "int8"]
    assert fields["network_type"]["disabled_modes"] == [settings.MINIMAX_H3_MODE]
    assert fields["compile"]["disabled_modes"] == [settings.MINIMAX_H3_MODE]


def test_cache_controls_have_plain_language_labels():
    schema = settings.settings_schema({"recache_latents": False, "recache_text": False})
    fields = {field["key"]: field for section in schema["sections"] for field in section["fields"]}

    assert fields["recache_latents"]["label"] == "Rebuild Image/Latent Cache"
    assert fields["recache_text"]["label"] == "Rebuild Caption/Text Cache"


def test_normal_cache_controls_are_in_guided_dataset_step_not_stages_panel():
    html = (Path(__file__).parents[1] / "modern_gui" / "static" / "index.html").read_text(encoding="utf-8")
    dataset_step = html.index('data-pane="data"')
    method_step = html.index('data-pane="method"')
    cache_card = html.index('id="normal-cache-policy"')
    plan_view = html.index('id="plan"')

    assert dataset_step < cache_card < method_step < plan_view
    assert html.count('id="normal-cache-policy"') == 1


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
