from pathlib import Path

from modern_gui.training_notes import effective_training_comment, training_settings_summary
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from musubi_tuner_gui import MusubiTunerGUI


def test_comment_defaults_on_in_both_interfaces_without_overwriting_note(monkeypatch, tmp_path):
    from modern_gui import settings as module
    path = tmp_path / "settings.json"
    monkeypatch.setattr(module, "LAST_SETTINGS", path)
    path.write_text(json.dumps({"training_comment": "Personal note."}))
    loaded = module.load_settings()
    assert loaded["auto_training_settings_summary"] is True
    for formatter in (effective_training_comment, MusubiTunerGUI._effective_training_comment):
        assert formatter({"training_comment": "Personal note."}).startswith("Personal note.\n\nSettings:")
        assert formatter({"training_comment": "Personal note.", "auto_training_settings_summary": False}) == "Personal note."
    module.save_settings({"training_comment": "Personal note.", "auto_training_settings_summary": False})
    assert module.load_settings()["auto_training_settings_summary"] is False


def test_summary_uses_actual_source_batches_and_current_stage_settings(tmp_path):
    dataset = tmp_path / "mixed.toml"
    dataset.write_text('[general]\nbatch_size=1\nresolution=[1024,1024]\n[[datasets]]\nimage_directory="one"\n[[datasets]]\nimage_directory="two"\nbatch_size=2\nresolution=[512,512]\n')
    settings = {"training_mode": "MOSS-Image (Experimental)", "dataset_config": str(dataset),
                "gradient_accumulation_steps": "4", "network_type": "LoKr", "lokr_factor": "4",
                "learning_rate": "0,0005", "lr_scheduler": "constant", "max_grad_norm": "1,0",
                "mixed_precision": "bf16", "stage_type": "standard", "use_staged_training": True,
                "max_train_steps": "120", "max_train_epochs": ""}
    summary = training_settings_summary(settings)
    for text in ("batch=1/2", "accum=4", "effective-batch≈4–8/GPU", "120 steps", "data=mixed (1024×1024/512×512)",
                 "LoKr factor 4", "lr=0.0005", "clip=1.0", "precision=bf16", "sched=constant"):
        assert text in summary
    assert "staged " not in summary
    assert summary == MusubiTunerGUI._training_settings_summary(settings)
    assert "batch=" not in training_settings_summary({"dataset_config": "missing.toml"})


def test_summary_records_active_moss_features_and_omits_unused_and_preview_features(tmp_path):
    settings = {"training_mode": "MOSS-Image (Experimental)", "network_type": "LoRA",
                "moss_repa_enabled": True, "moss_repa_weight": "0,5", "moss_repa_block": "8",
                "moss_ema_enabled": True, "moss_ema_decay": "0,98", "dop_enabled": True, "dop_loss_weight": "1",
                "moss_weight_noise_sigma": "0,0125", "moss_weight_noise_enabled": False,
                "moss_depth_anchor_enabled": True, "moss_depth_anchor_weight": "0,01", "moss_depth_anchor_vae_device": "secondary",
                "moss_fast_lora": True, "moss_unpadded_batches": True, "compile": True, "moss_compile_checkpoint": "outside",
                "moss_clean_end_share": "30:7:60", "moss_sample_lora": "preview-only-turbo.safetensors"}
    summary = training_settings_summary(settings)
    for text in ("REPA 0.5@layer8", "EMA 0.98", "DOP 1", "depth 0.01", "VAE=secondary", "clean-end=30:7:60", "BF16-LoRA", "unpadded", "compile/outside"):
        assert text in summary
    assert "weight-noise" not in summary and "preview-only-turbo" not in summary
    off = training_settings_summary({**settings, "moss_repa_enabled": False, "moss_ema_enabled": False, "moss_depth_anchor_enabled": False, "dop_enabled": False})
    assert all(text not in off for text in ("REPA", "EMA", "depth", "DOP"))


def test_h3_summary_describes_effective_quality_layers_and_native_limitations():
    settings = {"training_mode": "MiniMax H3 (Experimental)", "minimax_h3_training_assistant_enabled": True,
                "minimax_h3_dynamic_sigma_enabled": True, "minimax_h3_dynamic_sigma_every_n_steps": "5",
                "minimax_h3_guidance_distillation_sigma_min": "0,15", "minimax_h3_base_preservation_enabled": True,
                "krea2_depth_anchor_weight": "0.01", "dop_enabled": True}
    summary = training_settings_summary(settings)
    assert "Dynamic Sigma 4 σ≥0.15 every=5" in summary and "assistant=" in summary and "drift" in summary
    teacher = training_settings_summary({**settings, "minimax_h3_teacher_matching": True})
    assert "reference-learning=" in teacher and "Dynamic Sigma" not in teacher and "assistant=" not in teacher
    native = training_settings_summary({**settings, "minimax_h3_training_workflow": "Video + audio · official multimodal", "timestep_sampling": "krea2_shift"})
    assert "noise=linked uniform" in native and "every=1" in native
    assert all(text not in native for text in ("drift", "depth", "DOP", "krea2_shift"))


def test_preview_endpoint_uses_same_formatter_and_draft_without_writing_dataset(tmp_path):
    from modern_gui.server import MusubiWebHandler
    dataset = tmp_path / "dataset.toml"
    original = '[general]\nbatch_size=1\n[[datasets]]\nimage_directory="images"\n'
    dataset.write_text(original)
    settings = {"dataset_config": str(dataset), "gradient_accumulation_steps": "4"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        def preview(payload):
            request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/training-summary", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request) as response:
                return json.load(response)["summary"]
        assert preview({"settings": settings}) == training_settings_summary(settings)
        assert "effective-batch≈8/GPU" in preview({"settings": settings, "dataset_text": original.replace("batch_size=1", "batch_size=2")})
        assert dataset.read_text() == original
    finally:
        server.shutdown()
        server.server_close()


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
