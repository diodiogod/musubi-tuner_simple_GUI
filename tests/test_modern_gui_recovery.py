from pathlib import Path

import pytest

from modern_gui import recovery
from modern_gui.recovery import (
    effective_history_settings,
    import_output_jobs,
    prepare_continuation,
    prepare_exact_recovery,
    prepare_face_refinement,
    resolve_exact_recovery_state,
    validate_accelerate_state,
)


def test_legacy_minimax_replay_does_not_activate_new_depth_features():
    job = {
        "mode": "MiniMax H3 (Experimental)",
        "settings": {
            "training_mode": "MiniMax H3 (Experimental)",
            "krea2_generalization_preset": "Balanced Experimental",
            "krea2_weight_noise_sigma": "0.0125",
            "krea2_depth_anchor_weight": "0.01",
        },
    }

    settings = effective_history_settings(job)

    assert settings["krea2_generalization_preset"] == "Off (Baseline)"
    assert settings["krea2_weight_noise_sigma"] == "0"
    assert settings["krea2_depth_anchor_weight"] == "0"


def test_current_minimax_replay_preserves_explicit_depth_features():
    job = {
        "mode": "MiniMax H3 (Experimental)",
        "settings": {
            "training_mode": "MiniMax H3 (Experimental)",
            "minimax_h3_depth_every_n_steps": "2",
            "krea2_generalization_preset": "Balanced Experimental",
            "krea2_weight_noise_sigma": "0.0125",
            "krea2_depth_anchor_weight": "0.01",
        },
    }

    settings = effective_history_settings(job)

    assert settings["krea2_generalization_preset"] == "Balanced Experimental"
    assert settings["krea2_weight_noise_sigma"] == "0.0125"
    assert settings["krea2_depth_anchor_weight"] == "0.01"


def test_recorded_minimax_depth_command_is_not_reinterpreted_as_legacy_baseline():
    job = {
        "mode": "MiniMax H3 (Experimental)",
        "command": "train.py --depth_anchor_weight 0.01 --weight_noise_sigma 0.0125",
        "settings": {
            "training_mode": "MiniMax H3 (Experimental)",
            "krea2_generalization_preset": "Balanced Experimental",
            "krea2_weight_noise_sigma": "0.0125",
            "krea2_depth_anchor_weight": "0.01",
        },
    }

    settings = effective_history_settings(job)

    assert settings["krea2_generalization_preset"] == "Balanced Experimental"
    assert settings["krea2_depth_anchor_weight"] == "0.01"


def complete_state(path: Path) -> Path:
    path.mkdir(parents=True)
    for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
        (path / name).write_bytes(b"state")
    return path


def job_for(tmp_path: Path, state: Path, status="failed"):
    return {
        "kind": "training",
        "status": status,
        "output_name": "portrait",
        "current_epoch": 4,
        "settings_snapshot": {
            "training_mode": "Krea 2",
            "output_name": "portrait",
            "output_dir": str(tmp_path),
            "max_train_epochs": "10",
        },
        "resume_path": str(state),
    }


def test_exact_state_requires_all_accelerate_parts_and_position_marker(tmp_path: Path):
    complete = complete_state(tmp_path / "portrait-000004-state")
    valid, missing = validate_accelerate_state(complete)
    assert valid is True
    assert missing == []

    unnumbered = complete_state(tmp_path / "portrait-state")
    valid, missing = validate_accelerate_state(unnumbered)
    assert valid is False
    assert "epoch/step position marker in the state-folder name" in missing


def test_true_recovery_preserves_identity_and_enables_exact_position(tmp_path: Path):
    state = complete_state(tmp_path / "portrait-000004-state")

    settings = prepare_exact_recovery(job_for(tmp_path, state))

    assert settings["output_name"] == "portrait"
    assert settings["resume_path"] == str(state)
    assert settings["starting_point_mode"] == "state"
    assert settings["resume_exact_position"] is True
    assert settings["recovery_mode"] is True
    assert settings["recache_latents"] is False
    assert settings["use_staged_training"] is False


def test_completed_run_can_be_extended_from_its_final_positional_state(tmp_path: Path):
    state = complete_state(tmp_path / "portrait-000005-state")
    job = job_for(tmp_path, state, status="completed")
    job["current_epoch"] = 5
    job["settings_snapshot"]["max_train_epochs"] = "5"

    settings = prepare_exact_recovery(job)

    assert settings["output_name"] == "portrait"
    assert settings["resume_path"] == str(state)
    assert settings["resume_exact_position"] is True
    assert settings["recovery_mode"] is True


def test_additive_continuation_uses_saved_adapter_weights_and_never_claims_exact_position(tmp_path: Path):
    state = complete_state(tmp_path / "portrait-000004-state")

    settings = prepare_continuation(job_for(tmp_path, state, status="completed"))

    assert settings["output_name"] == "portrait-cont"
    assert settings["resume_path"] == ""
    assert settings["network_weights"] == str(state / "model.safetensors")
    assert settings["starting_point_mode"] == "weights"
    assert settings["resume_exact_position"] is False
    assert settings["recovery_mode"] is False


def test_staged_continuation_uses_final_stage_recipe_and_saved_adapter_weights(tmp_path: Path):
    base_state = complete_state(tmp_path / "portrait" / "portrait-000010-state")
    final_state = complete_state(
        tmp_path / "portrait-1024px" / "portrait-1024px-000002-state"
    )
    base_settings = {
        "training_mode": "Krea 2",
        "output_name": "portrait",
        "output_dir": str(tmp_path),
        "dataset_config": "base.toml",
        "max_train_epochs": "10",
        "sample_prompts_data": [{"prompt": "portrait of subject", "seed": 7}],
        "use_staged_training": True,
        "staged_training_config": [
            {
                "label": "1024",
                "enabled": True,
                "dataset_config": "final.toml",
                "epochs": "2",
            }
        ],
    }
    job = {
        "kind": "staged_training",
        "status": "completed",
        "settings": base_settings,
        "resume_path": str(base_state),
        "final_stage_settings": {
            **base_settings,
            "stage_type": "standard",
            "output_name": "portrait-1024px",
            "dataset_config": "final.toml",
            "max_train_epochs": "2",
            "max_train_steps": "",
        },
        "final_stage_artifacts": {
            "state": str(final_state),
            "lora": "",
        },
    }

    settings = prepare_continuation(job)

    assert settings["output_name"] == "portrait-1024px-cont"
    assert settings["dataset_config"] == "final.toml"
    assert settings["max_train_epochs"] == "2"
    assert settings["sample_prompts_data"] == [{"prompt": "portrait of subject", "seed": 7}]
    assert settings["resume_path"] == ""
    assert settings["network_weights"] == str(final_state / "model.safetensors")
    assert settings["starting_point_mode"] == "weights"
    assert settings["use_staged_training"] is False
    assert settings["resume_exact_position"] is False
    assert settings["recache_latents"] is False
    assert "stage_type" not in settings
    assert base_settings["output_name"] == "portrait"
    assert base_settings["dataset_config"] == "base.toml"


def test_face_final_staged_continuation_uses_completed_lora_weights(tmp_path: Path):
    base_state = complete_state(tmp_path / "portrait" / "portrait-000010-state")
    final_lora = (
        tmp_path
        / "portrait-face-pass"
        / "portrait-face-pass.safetensors"
    )
    final_lora.parent.mkdir()
    final_lora.write_bytes(b"refined lora")
    base_settings = {
        "training_mode": "Krea 2",
        "output_name": "portrait",
        "output_dir": str(tmp_path),
        "dataset_config": "base.toml",
        "max_train_epochs": "10",
        "use_staged_training": True,
        "staged_training_config": [
            {
                "label": "face-pass",
                "type": "face_refinement",
                "enabled": True,
                "steps": "30",
            }
        ],
    }
    job = {
        "kind": "staged_training",
        "status": "completed",
        "settings": base_settings,
        "resume_path": str(base_state),
        "final_stage_settings": {
            **base_settings,
            "stage_type": "face_refinement",
            "output_name": "portrait-face-pass",
            "max_train_epochs": "",
            "max_train_steps": "30",
            "face_output_path": str(tmp_path / "missing-refined.safetensors"),
        },
        "final_stage_artifacts": {
            "state": "",
            "lora": str(final_lora),
        },
    }

    settings = prepare_continuation(job)

    assert settings["output_name"] == "portrait-face-pass-cont"
    assert settings["max_train_steps"] == "30"
    assert settings["network_weights"] == str(final_lora)
    assert settings["resume_path"] == ""
    assert settings["starting_point_mode"] == "weights"
    assert settings["resume_exact_position"] is False
    assert settings["recovery_mode"] is False
    assert settings["use_staged_training"] is False
    assert "stage_type" not in settings
    assert "face_output_path" not in settings


def test_legacy_face_final_staged_continuation_finds_derived_lora(tmp_path: Path):
    final_lora = (
        tmp_path
        / "portrait-identity"
        / "portrait-identity.safetensors"
    )
    final_lora.parent.mkdir()
    final_lora.write_bytes(b"legacy refined lora")
    job = {
        "kind": "staged_training",
        "status": "completed",
        "settings": {
            "training_mode": "Krea 2",
            "output_name": "portrait",
            "output_dir": str(tmp_path),
            "dataset_config": "base.toml",
            "max_train_epochs": "10",
            "use_staged_training": True,
            "staged_training_config": [
                {
                    "label": "base",
                    "type": "standard",
                    "enabled": True,
                    "dataset_config": "base.toml",
                    "epochs": "1",
                },
                {
                    "label": "identity",
                    "type": "face_refinement",
                    "enabled": True,
                    "steps": "20",
                },
            ],
        },
    }

    settings = prepare_continuation(job)

    assert settings["output_name"] == "portrait-identity-cont"
    assert settings["max_train_steps"] == "20"
    assert settings["network_weights"] == str(final_lora)
    assert settings["resume_path"] == ""
    assert settings["starting_point_mode"] == "weights"


def test_incomplete_state_is_rejected_for_true_recovery(tmp_path: Path):
    state = tmp_path / "portrait-000004-state"
    state.mkdir()
    (state / "model.safetensors").write_bytes(b"state")
    job = job_for(tmp_path, state)

    resolved, rejected = resolve_exact_recovery_state(job)

    assert resolved is None
    assert any("optimizer state" in item for item in rejected)
    with pytest.raises(ValueError, match="No complete positional"):
        prepare_exact_recovery(job)


def test_import_output_jobs_adds_unrecorded_run_folders(monkeypatch, tmp_path: Path):
    history = tmp_path / "history.json"
    monkeypatch.setattr(recovery, "DESKTOP_HISTORY_PATH", history)
    output = tmp_path / "models"
    run = output / "portrait"
    run.mkdir(parents=True)
    (run / "portrait.safetensors").write_bytes(b"lora")

    result = import_output_jobs(
        {"training_mode": "Krea 2", "output_dir": str(output), "output_name": "current"}
    )

    assert result["added"] == 1
    imported = recovery.load_desktop_history()
    assert imported[0]["output_name"] == "portrait"
    assert imported[0]["status"] == "discovered"


def test_face_refinement_from_krea_history_resolves_lora_and_builds_typed_stage(tmp_path: Path):
    import torch
    from safetensors.torch import save_file

    run = tmp_path / "portrait"
    run.mkdir()
    lora = run / "portrait.safetensors"
    prefix = "lora_unet_blocks_0_attn_wq"
    save_file(
        {
            f"{prefix}.lora_down.weight": torch.zeros(2, 4),
            f"{prefix}.lora_up.weight": torch.zeros(4, 2),
        },
        lora,
    )
    job = {
        "kind": "training",
        "status": "completed",
        "settings_snapshot": {
            "training_mode": "Krea 2",
            "output_name": "portrait",
            "output_dir": str(tmp_path),
            "max_train_epochs": "",
        },
    }

    settings = prepare_face_refinement(job)

    assert settings["face_refinement_config"]["input_lora"] == str(lora.resolve())
    assert settings["face_refinement_config"]["input_mode"] == "existing_lora"
    assert settings["use_staged_training"] is True
    assert settings["staged_training_config"][0]["type"] == "face_refinement"
