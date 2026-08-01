from pathlib import Path

import pytest

from modern_gui.stages import (
    candidate_lora_paths,
    candidate_state_paths,
    prepare_standard_stage,
    resolve_stage_lora,
    resolve_standard_state,
    stage_label,
    validate_stage_plan,
)


def test_prepare_standard_stage_applies_limits_overrides_and_additive_resume(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    base = {
        "training_mode": "Krea 2",
        "output_name": "portrait",
        "output_dir": str(tmp_path),
        "dop_enabled": False,
        "staged_recache_latents": True,
        "staged_recache_text": False,
    }
    stage = {
        "label": "1024",
        "dataset_config": str(dataset),
        "steps": "250",
        "epochs": "",
        "dop_mode": "enable",
        "dop_loss_weight": "0.4",
        "dop_trigger_word": "sks",
        "dop_class_word": "person",
        "depth_helpers_mode": "keep on GPU",
    }

    prepared = prepare_standard_stage(base, stage, 1, resume_path="previous-state")

    assert prepared["output_name"] == "portrait-1024px"
    assert prepared["max_train_steps"] == "250"
    assert prepared["max_train_epochs"] == ""
    assert prepared["dop_enabled"] is True
    assert prepared["dop_loss_weight"] == "0.4"
    assert prepared["krea2_keep_depth_helpers_on_gpu"] is True
    assert prepared["resume_path"] == "previous-state"
    assert prepared["network_weights"] == ""
    assert prepared["resume_exact_position"] is False
    assert prepared["recovery_mode"] is False


def test_first_standard_stage_preserves_base_lora_continuation(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    weights = tmp_path / "existing.safetensors"
    weights.write_bytes(b"lora")

    prepared = prepare_standard_stage(
        {
            "training_mode": "Krea 2",
            "output_name": "portrait",
            "output_dir": str(tmp_path),
            "network_weights": str(weights),
        },
        {
            "enabled": True,
            "label": "first",
            "dataset_config": str(dataset),
            "epochs": "1",
        },
        0,
    )

    assert prepared["network_weights"] == str(weights)


def test_validate_plan_rejects_separate_wan_runs(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot map one state"):
        validate_stage_plan(
            {
                "training_mode": "Wan 2.2",
                "train_low_noise": True,
                "train_high_noise": True,
                "network_dim_high": "16",
                "staged_training_config": [
                    {"enabled": True, "dataset_config": str(dataset), "epochs": "1"}
                ],
            }
        )


def test_artifact_resolution_accepts_numbered_or_legacy_names(tmp_path: Path):
    settings = {
        "training_mode": "Krea 2",
        "output_name": "run-512px",
        "output_dir": str(tmp_path),
        "max_train_epochs": "3",
    }
    numbered_state, legacy_state = candidate_state_paths(settings)
    numbered_lora, legacy_lora = candidate_lora_paths(settings)
    legacy_state.mkdir(parents=True)
    legacy_lora.write_bytes(b"lora")

    assert resolve_standard_state(settings) == legacy_state
    assert resolve_stage_lora(settings) == legacy_lora
    assert numbered_state.name.endswith("-000003-state")
    assert numbered_lora.name.endswith("-000003.safetensors")


def test_stage_label_is_windows_path_safe():
    assert stage_label({"label": 'face: left/right?'}) == "face- left-right"


def test_validate_plan_rejects_duplicate_or_unsafe_artifact_labels(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    base = {
        "training_mode": "Krea 2",
        "staged_training_config": [
            {"enabled": True, "label": "Portrait", "dataset_config": str(dataset), "epochs": "1"},
            {"enabled": True, "label": "portrait", "dataset_config": str(dataset), "epochs": "1"},
        ],
    }

    with pytest.raises(ValueError, match="must be unique"):
        validate_stage_plan(base)

    base["staged_training_config"][1]["label"] = "bad/name"
    with pytest.raises(ValueError, match="not safe"):
        validate_stage_plan(base)


def test_stage_without_explicit_enabled_flag_remains_included(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")

    stages = validate_stage_plan(
        {
            "training_mode": "Krea 2",
            "staged_training_config": [
                {"label": "one", "dataset_config": str(dataset), "epochs": "1"}
            ],
        }
    )

    assert stages[0]["label"] == "one"


def test_validate_plan_rejects_unsupported_or_invalid_stage_dop(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    stage = {
        "enabled": True,
        "label": "one",
        "dataset_config": str(dataset),
        "epochs": "1",
        "dop_mode": "enable",
        "dop_loss_weight": "1",
        "dop_trigger_word": "sks",
        "dop_class_word": "person",
    }

    with pytest.raises(ValueError, match="supported only"):
        validate_stage_plan(
            {"training_mode": "Wan 2.2", "staged_training_config": [stage]}
        )

    stage["dop_loss_weight"] = "0"
    with pytest.raises(ValueError, match="DOP configuration"):
        validate_stage_plan(
            {"training_mode": "Krea 2", "staged_training_config": [stage]}
        )


def test_first_face_stage_requires_existing_lora_mode_and_file(tmp_path: Path):
    stage = {
        "enabled": True,
        "label": "face",
        "type": "face_refinement",
        "steps": "10",
    }
    settings = {
        "training_mode": "Krea 2",
        "staged_training_config": [stage],
        "face_refinement_config": {"input_mode": "previous_stage"},
    }

    with pytest.raises(ValueError, match="first enabled stage"):
        validate_stage_plan(settings)

    settings["face_refinement_config"] = {
        "input_mode": "existing_lora",
        "input_lora": str(tmp_path / "missing.safetensors"),
    }
    with pytest.raises(ValueError, match="does not exist"):
        validate_stage_plan(settings)
