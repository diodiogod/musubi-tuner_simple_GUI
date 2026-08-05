from pathlib import Path

from modern_gui.validation import validate_training_settings


def valid_krea_settings(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    return {
        "training_mode": "Krea 2",
        "dataset_config": str(dataset),
        "output_dir": str(tmp_path),
        "output_name": "portrait",
        "vae_model": "vae.safetensors",
        "krea2_dit_model": "raw.safetensors",
        "krea2_text_encoder": "text",
        "learning_rate": "2e-4",
        "gradient_accumulation_steps": "1",
        "network_dim_low": "32",
        "max_train_epochs": "2",
    }


def test_valid_krea_setup_passes_preflight(tmp_path: Path):
    result = validate_training_settings(valid_krea_settings(tmp_path))

    assert result["errors"] == []


def test_off_generalization_preset_rejects_stale_nonzero_depth(tmp_path: Path):
    settings = valid_krea_settings(tmp_path)
    settings.update(
        {
            "krea2_generalization_preset": "Off (Baseline)",
            "krea2_weight_noise_sigma": "0",
            "krea2_depth_anchor_weight": "0.01",
        }
    )

    messages = [item["message"] for item in validate_training_settings(settings)["errors"]]

    assert any("preset is Off" in message for message in messages)


def test_krea_incompatible_runtime_options_are_actionable(tmp_path: Path):
    settings = valid_krea_settings(tmp_path)
    settings.update(
        {
            "fp8_base": True,
            "fp8_scaled": False,
            "krea2_turbo_dit": "turbo.safetensors",
            "blocks_to_swap": "8",
        }
    )

    result = validate_training_settings(settings)
    messages = [item["message"] for item in result["errors"]]

    assert "Krea 2 FP8 Base requires FP8 Scaled." in messages
    assert "Krea Turbo sampling cannot be combined with Blocks to Swap." in messages


def test_minimax_h3_preflight_enforces_experimental_24gb_contract(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "dataset_config": str(dataset),
        "output_dir": str(tmp_path),
        "output_name": "h3",
        "vae_model": "vae.safetensors",
        "minimax_h3_dit_model": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "learning_rate": "2e-4",
        "gradient_accumulation_steps": "1",
        "network_dim_low": "16",
        "network_type": "LoRA",
        "max_train_epochs": "1",
        "blocks_to_swap": "30",
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
    }

    result = validate_training_settings(settings)
    assert result["errors"] == []
    assert any("two-epoch run" in item["message"] for item in result["warnings"])

    settings.update({"compile": True, "fp8_base": True, "blocks_to_swap": "0", "sample_at_first": True})
    messages = [item["message"] for item in validate_training_settings(settings)["errors"]]
    assert any("Torch Compile" in message for message in messages)
    assert any("FP8 flags" in message for message in messages)
    assert any("1–48" in message for message in messages)
    assert any("compact Qwen3-VL text encoder" in message for message in messages)


def test_minimax_h3_dop_and_training_samples_are_accepted(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "dataset_config": str(dataset),
        "output_dir": str(tmp_path),
        "output_name": "h3",
        "vae_model": "vae.safetensors",
        "minimax_h3_dit_model": "dit.safetensors",
        "minimax_h3_text_encoder": "te.safetensors",
        "learning_rate": "1e-4",
        "gradient_accumulation_steps": "1",
        "network_dim_low": "16",
        "network_type": "LoRA",
        "max_train_epochs": "1",
        "blocks_to_swap": "30",
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
        "sample_every_n_epochs": "1",
        "sample_prompts_data": [{"enabled": True, "prompt": "portrait", "width": 768, "height": 768}],
        "dop_enabled": True,
        "dop_loss_weight": "0.2",
        "dop_trigger_word": "sks",
        "dop_class_word": "person",
    }
    result = validate_training_settings(settings)
    assert result["errors"] == []

    settings["sample_prompts_data"][0].update({"neg": "blur", "guidance": 5, "width": 770})
    messages = [item["message"] for item in validate_training_settings(settings)["errors"]]
    assert any("negative prompt" in message for message in messages)
    assert any("guidance and CFG at 1.0" in message for message in messages)
    assert any("multiples of 32" in message for message in messages)


def test_minimax_secondary_depth_configuration_is_validated(tmp_path: Path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "dataset_config": str(dataset),
        "output_dir": str(tmp_path),
        "output_name": "h3-depth",
        "vae_model": "vae.safetensors",
        "minimax_h3_dit_model": "dit.safetensors",
        "learning_rate": "1e-4",
        "gradient_accumulation_steps": "1",
        "network_dim_low": "16",
        "network_type": "LoRA",
        "max_train_epochs": "1",
        "blocks_to_swap": "30",
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
        "krea2_generalization_preset": "Balanced Experimental",
        "krea2_depth_anchor_weight": "0.01",
        "minimax_h3_depth_vae_device": "secondary",
        "minimax_h3_keep_depth_vae_on_device": True,
        "minimax_h3_depth_every_n_steps": "4",
    }

    assert validate_training_settings(settings)["errors"] == []

    settings["minimax_h3_depth_every_n_steps"] = "0"
    messages = [item["message"] for item in validate_training_settings(settings)["errors"]]
    assert any("cadence" in message for message in messages)


def test_exact_resume_is_revalidated_even_when_loaded_from_json(tmp_path: Path):
    settings = valid_krea_settings(tmp_path)
    state = tmp_path / "portrait-000002-state"
    state.mkdir()
    (state / "model.safetensors").write_bytes(b"only model")
    settings.update({"resume_path": str(state), "resume_exact_position": True})

    result = validate_training_settings(settings)

    assert any("Exact recovery state is incomplete" in item["message"] for item in result["errors"])


def test_resume_and_weight_continuation_are_mutually_exclusive(tmp_path: Path):
    settings = valid_krea_settings(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"weights")
    settings.update({"resume_path": str(state), "network_weights": str(weights)})

    result = validate_training_settings(settings)

    assert any("mutually exclusive" in item["message"] for item in result["errors"])


def test_staged_plan_errors_are_included_in_normal_preflight(tmp_path: Path):
    settings = valid_krea_settings(tmp_path)
    settings.update(
        {
            "use_staged_training": True,
            "staged_training_config": [
                {"enabled": True, "label": "broken", "dataset_config": "", "epochs": "1"}
            ],
        }
    )

    result = validate_training_settings(settings)

    assert any(item["key"] == "staged_training_config" for item in result["errors"])


def test_included_sample_prompts_are_validated_and_empty_schedule_is_explained(tmp_path: Path):
    settings = valid_krea_settings(tmp_path)
    settings.update(
        {
            "sample_every_n_epochs": "1",
            "sample_prompts_data": [
                {"enabled": True, "prompt": "", "width": "wide"},
            ],
        }
    )

    result = validate_training_settings(settings)

    assert any("has no positive prompt" in item["message"] for item in result["errors"])
    assert any("width must be a positive whole number" in item["message"] for item in result["errors"])

    settings["sample_prompts_data"][0]["enabled"] = False
    result = validate_training_settings(settings)
    assert not any(item["key"] == "sample_prompts_data" for item in result["errors"])
    assert any("no sample prompt is included" in item["message"] for item in result["warnings"])


def test_saved_state_recovery_is_not_silently_ignored_by_staged_plan(tmp_path: Path):
    settings = valid_krea_settings(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    dataset = tmp_path / "stage.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    settings.update(
        {
            "resume_path": str(state),
            "use_staged_training": True,
            "staged_training_config": [
                {
                    "enabled": True,
                    "label": "first",
                    "dataset_config": str(dataset),
                    "epochs": "1",
                }
            ],
        }
    )

    result = validate_training_settings(settings)

    assert any("cannot be combined with a staged plan" in item["message"] for item in result["errors"])


def test_sampling_cadence_rejects_values_argparse_cannot_use(tmp_path: Path):
    settings = valid_krea_settings(tmp_path)
    settings.update({"sample_every_n_epochs": "1.5", "sample_every_n_steps": "-2"})

    result = validate_training_settings(settings)

    assert any(item["key"] == "sample_every_n_epochs" for item in result["errors"])
    assert any(item["key"] == "sample_every_n_steps" for item in result["errors"])


def test_existing_lora_face_only_plan_skips_irrelevant_sft_requirements(tmp_path: Path):
    lora = tmp_path / "identity.safetensors"
    lora.write_bytes(b"lora")
    settings = valid_krea_settings(tmp_path)
    settings.pop("dataset_config")
    settings.pop("learning_rate")
    settings.pop("network_dim_low")
    settings.update(
        {
            "use_staged_training": True,
            "staged_training_config": [
                {
                    "enabled": True,
                    "label": "face",
                    "type": "face_refinement",
                    "steps": "10",
                }
            ],
            "face_refinement_config": {
                "input_mode": "existing_lora",
                "input_lora": str(lora),
            },
        }
    )

    result = validate_training_settings(settings)

    assert not any(
        item["key"] in {"dataset_config", "learning_rate", "network_dim_low"}
        for item in result["errors"]
    )
