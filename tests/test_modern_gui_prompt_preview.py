from pathlib import Path

import pytest

from modern_gui.prompt_preview import build_krea_preview, build_minimax_h3_preview, serialize_prompt
from modern_gui.sample_prompts import serialize_sample_prompt


def test_modern_plan_uses_full_preview_dialog_hook_for_prompt_cards():
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")
    assert 'visual.addEventListener("dblclick"' in app_js
    assert 'openSamplePreview({' in app_js
    assert 'url:previewUrl' in app_js


def test_minimax_face_runtime_group_opens_for_preview_controls():
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")
    assert '"Runtime and checkpoints"' in app_js
    assert '],isH3);' in app_js


def test_modern_face_pose_plan_uses_comparison_table():
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")
    index_html = Path("modern_gui/static/index.html").read_text(encoding="utf-8")
    assert "function renderPosePlanTable" in app_js
    assert 'role="table" aria-label="Pose training plan"' in app_js
    assert "data-pose-field" in app_js
    assert "data-pose-prompts" in app_js
    assert "Add suggested prompts" in app_js
    assert "pose-prompt-tabs" in app_js
    assert 'id="face-fallback-prompts"' in index_html
    assert 'id="pose-plan-help"' in index_html


def test_modern_run_can_persist_split_layout_and_terminal_theme():
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")
    index_html = Path("modern_gui/static/index.html").read_text(encoding="utf-8")
    styles_css = Path("modern_gui/static/styles.css").read_text(encoding="utf-8")
    assert 'id="toggle-run-split"' in index_html
    assert 'id="run-split-divider"' in index_html
    assert 'readLocalPreference("musubi-run-split"' in app_js
    assert 'writeLocalPreference("musubi-log-follow"' in app_js
    assert "function keepLiveLogAtBottom" in app_js
    assert 'data-run-tab=log' in app_js
    assert "--terminal-bg" in styles_css
    assert "#run.run-split-view .run-panel-stack" in styles_css


def test_live_log_reenables_follow_when_scrolled_back_to_bottom():
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")

    assert "function isLiveLogAtBottom" in app_js
    assert "function setFollowLog" in app_js
    assert "if(isLiveLogAtBottom(log)){if(!followLog)setFollowLog(true)}" in app_js
    assert '$("#follow-log").addEventListener("click",()=>setFollowLog(!followLog' in app_js


def test_run_tab_click_exits_split_view():
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")

    assert 'if($("#run").classList.contains("run-split-view"))setRunSplitView(false)' in app_js


def test_repeat_names_use_numbered_canonical_suffixes():
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")

    assert "function splitRepeatName" in app_js
    assert "function nextRepeatName" in app_js
    assert "snapshot.output_name=nextRepeatName" in app_js


def test_shared_history_exposes_performance_log_and_comparison_ui():
    index_html = Path("modern_gui/static/index.html").read_text(encoding="utf-8")
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")

    assert 'id="job-view-console"' in index_html
    assert 'id="compare-jobs"' in index_html
    assert 'id="job-compare-dialog"' in index_html
    assert "function drawJobSpeedChart" in app_js
    assert "function toggleJobComparison" in app_js
    assert "function replaySettings" in app_js


def test_advanced_training_help_is_specific_and_dop_is_separate():
    index_html = Path("modern_gui/static/index.html").read_text(encoding="utf-8")
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")

    assert 'id="dop-settings"' in index_html
    assert 'id="dop-fields"' in index_html
    assert "Differential Output Preservation (DoP)" in index_html
    assert "Weight noise and structural depth" in index_html
    assert "krea2_keep_depth_helpers_on_gpu:" in app_js
    assert "krea2_depth_anchor_grad_checkpoint:" in app_js
    assert "preserved in saved recipes and job history" not in app_js
    assert 'id="minimax-depth-hardware-notice"' in index_html
    assert "function renderMinimaxDepthHardwareNotice" in app_js
    assert "16 GB VRAM" in app_js
    assert "8 GB helper GPU is not supported" in app_js
    assert 'id="krea-depth-compute"' in index_html
    assert "krea2_depth_vae_device:" in app_js


def test_history_recipe_restore_loads_and_validates_saved_dataset_toml():
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")

    assert "async function loadDatasetForSettings" in app_js
    assert "const datasetLoaded=await loadDatasetForSettings()" in app_js
    assert "The Dataset TOML could not be loaded" in app_js
    assert "if (state.settings.dataset_config && !await loadDatasetForSettings())" in app_js


def test_modern_theme_preference_overrides_stale_saved_recipe_on_startup():
    app_js = Path("modern_gui/static/app.js").read_text(encoding="utf-8")

    assert 'const savedTheme=readLocalPreference("musubi-theme",state.settings.appearance_mode||"Dark")' in app_js
    assert 'applyTheme(savedTheme,{syncSetting:true})' in app_js


def test_serialize_krea_preview_prompt():
    assert serialize_prompt({"prompt": "portrait", "width": 512, "seed": 42, "neg": "blur"}) == (
        "portrait --w 512 --d 42 --n blur"
    )


def test_serialize_training_prompt_preserves_mode_specific_flags():
    prompt = {
        "prompt": "motion",
        "width": "832",
        "height": "480",
        "steps": "20",
        "guidance": "5",
        "frames": "25",
        "flow_shift": "3",
        "cfg_scale": "1",
        "seed": "0",
        "neg": "blur",
        "image_path": "start.png",
    }

    assert serialize_sample_prompt(prompt, "Wan 2.2") == (
        "motion --w 832 --h 480 --s 20 --g 5 --f 25 --fs 3 "
        "--l 1 --d 0 --n blur --i start.png"
    )


def test_build_krea_preview_uses_turbo_and_batch_file(tmp_path: Path):
    models = {}
    for name in ("raw.safetensors", "turbo.safetensors", "vae.safetensors", "text.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        models[name] = str(path)
    command, save_path = build_krea_preview(
        {
            "training_mode": "Krea 2",
            "krea2_dit_model": models["raw.safetensors"],
            "krea2_turbo_dit": models["turbo.safetensors"],
            "vae_model": models["vae.safetensors"],
            "krea2_text_encoder": models["text.safetensors"],
            "output_dir": str(tmp_path),
            "output_name": "preview",
            "attention_mechanism": "sdpa",
        },
        [{"prompt": "one", "enabled": True}, {"prompt": "two", "enabled": True}],
    )

    assert "--turbo" in command
    assert "--from_file" in command
    assert save_path == tmp_path / "preview" / "sample_test"
    prompt_file = Path(command[command.index("--from_file") + 1])
    assert prompt_file.parent == save_path
    assert prompt_file.name.startswith("preview_prompts_")
    assert prompt_file.read_text(encoding="utf-8") == "one\ntwo"
    assert not list(save_path.glob(".preview-prompts-*.tmp"))

    second_command, _ = build_krea_preview(
        {
            "training_mode": "Krea 2",
            "krea2_dit_model": models["raw.safetensors"],
            "krea2_turbo_dit": models["turbo.safetensors"],
            "vae_model": models["vae.safetensors"],
            "krea2_text_encoder": models["text.safetensors"],
            "output_dir": str(tmp_path),
            "output_name": "preview",
            "attention_mechanism": "sdpa",
        },
        [{"prompt": "replacement", "enabled": True}, {"prompt": "batch", "enabled": True}],
    )
    second_prompt_file = Path(second_command[second_command.index("--from_file") + 1])
    assert second_prompt_file != prompt_file
    assert prompt_file.read_text(encoding="utf-8") == "one\ntwo"
    assert second_prompt_file.read_text(encoding="utf-8") == "replacement\nbatch"


def test_build_minimax_h3_preview_uses_compact_models_and_latest_lora(tmp_path: Path):
    paths = {}
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths[name] = str(path)
    run = tmp_path / "portrait"
    run.mkdir()
    lora = run / "portrait-000002.safetensors"
    lora.write_bytes(b"lora")
    command, save_path = build_minimax_h3_preview(
        {
            "training_mode": "MiniMax H3 (Experimental)",
            "minimax_h3_dit_model": paths["dit.safetensors"],
            "vae_model": paths["vae.safetensors"],
            "minimax_h3_text_encoder": paths["te.safetensors"],
            "output_dir": str(tmp_path),
            "output_name": "portrait",
            "blocks_to_swap": "30",
            "attention_mechanism": "sdpa",
        },
        [{"prompt": "portrait", "width": 768, "height": 768, "guidance": 1}],
    )
    assert command[1].endswith("minimax_h3_video_generate.py")
    assert command[command.index("--network_weights") + 1] == str(lora)
    assert command[command.index("--lora_multiplier") + 1] == "1.0"
    assert command[command.index("--video_shift") + 1] == "12.0"
    assert command[command.index("--frames") + 1] == "39"
    assert command[command.index("--steps") + 1] == "20"
    assert Path(command[command.index("--output") + 1]).suffix == ".mp4"
    assert save_path == run / "sample_test"


def test_build_minimax_h3_preview_finds_lora_when_output_dir_is_run_folder(tmp_path: Path):
    paths = {}
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths[name] = str(path)
    run = tmp_path / "portrait-run"
    run.mkdir()
    lora = run / "portrait-000002.safetensors"
    lora.write_bytes(b"lora")

    command, _ = build_minimax_h3_preview(
        {
            "training_mode": "MiniMax H3 (Experimental)",
            "minimax_h3_dit_model": paths["dit.safetensors"],
            "vae_model": paths["vae.safetensors"],
            "minimax_h3_text_encoder": paths["te.safetensors"],
            "output_dir": str(run),
            "output_name": "portrait",
            "preview_use_lora": True,
        },
        [{"prompt": "portrait", "width": 768, "height": 768, "guidance": 1}],
    )

    assert command[command.index("--network_weights") + 1] == str(lora)


def test_build_minimax_h3_preview_can_explicitly_disable_lora(tmp_path: Path):
    paths = {}
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths[name] = str(path)
    run = tmp_path / "portrait"
    run.mkdir()
    (run / "portrait-000002.safetensors").write_bytes(b"lora")

    command, _ = build_minimax_h3_preview(
        {
            "training_mode": "MiniMax H3 (Experimental)",
            "minimax_h3_dit_model": paths["dit.safetensors"],
            "vae_model": paths["vae.safetensors"],
            "minimax_h3_text_encoder": paths["te.safetensors"],
            "output_dir": str(tmp_path),
            "output_name": "portrait",
            "preview_use_lora": False,
        },
        [{"prompt": "portrait", "width": 768, "height": 768, "guidance": 1}],
    )

    assert "--network_weights" not in command


def test_build_minimax_h3_preview_uses_selected_lora_and_strength(tmp_path: Path):
    paths = {}
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors", "identity.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths[name] = str(path)

    command, _ = build_minimax_h3_preview(
        {
            "training_mode": "MiniMax H3 (Experimental)",
            "minimax_h3_dit_model": paths["dit.safetensors"],
            "vae_model": paths["vae.safetensors"],
            "minimax_h3_text_encoder": paths["te.safetensors"],
            "output_dir": str(tmp_path),
            "preview_use_lora": True,
            "preview_lora_path": paths["identity.safetensors"],
            "preview_lora_multiplier": "0.75",
        },
        [{"prompt": "portrait", "width": 768, "height": 768, "guidance": 1}],
    )

    assert command[command.index("--network_weights") + 1] == paths["identity.safetensors"]
    assert command[command.index("--lora_multiplier") + 1] == "0.75"


def test_build_minimax_h3_preview_uses_selected_resume_state_model(tmp_path: Path):
    paths = {}
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths[name] = str(path)
    run = tmp_path / "portrait"
    run.mkdir()
    older = run / "portrait-000001.safetensors"
    older.write_bytes(b"older")
    state = run / "portrait-state"
    state.mkdir()
    resumed = state / "model.safetensors"
    resumed.write_bytes(b"completed")

    command, _ = build_minimax_h3_preview(
        {
            "training_mode": "MiniMax H3 (Experimental)",
            "minimax_h3_dit_model": paths["dit.safetensors"],
            "vae_model": paths["vae.safetensors"],
            "minimax_h3_text_encoder": paths["te.safetensors"],
            "output_dir": str(tmp_path),
            "output_name": "portrait",
            "starting_point_mode": "state",
            "resume_path": str(state),
            "preview_use_lora": True,
        },
        [{"prompt": "portrait", "guidance": 1}],
    )

    assert command[command.index("--network_weights") + 1] == str(resumed)


def test_build_minimax_h3_preview_does_not_fallback_from_broken_resume_state(tmp_path: Path):
    paths = {}
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths[name] = str(path)
    run = tmp_path / "portrait"
    run.mkdir()
    (run / "portrait-000001.safetensors").write_bytes(b"older")
    state = run / "portrait-state"
    state.mkdir()
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "minimax_h3_dit_model": paths["dit.safetensors"],
        "vae_model": paths["vae.safetensors"],
        "minimax_h3_text_encoder": paths["te.safetensors"],
        "output_dir": str(tmp_path),
        "output_name": "portrait",
        "starting_point_mode": "state",
        "resume_path": str(state),
        "preview_use_lora": True,
    }

    with pytest.raises(ValueError, match="no model.safetensors"):
        build_minimax_h3_preview(settings, [{"prompt": "portrait", "guidance": 1}])


def test_build_minimax_h3_preview_rejects_cfg_and_batch(tmp_path: Path):
    paths = []
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(str(path))
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "minimax_h3_dit_model": paths[0],
        "vae_model": paths[1],
        "minimax_h3_text_encoder": paths[2],
        "output_dir": str(tmp_path),
    }
    with pytest.raises(ValueError, match="one prompt at a time"):
        build_minimax_h3_preview(settings, [{"prompt": "one"}, {"prompt": "two"}])
    with pytest.raises(ValueError, match="negative prompts"):
        build_minimax_h3_preview(settings, [{"prompt": "one", "neg": "blur"}])


def test_build_minimax_h3_preview_keeps_explicit_legacy_still(tmp_path: Path):
    paths = []
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(str(path))
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "minimax_h3_dit_model": paths[0],
        "vae_model": paths[1],
        "minimax_h3_text_encoder": paths[2],
        "output_dir": str(tmp_path),
        "preview_use_lora": False,
    }

    command, _ = build_minimax_h3_preview(settings, [{"prompt": "one", "frames": 1}])

    assert command[1].endswith("minimax_h3_image_generate.py")
    assert "--frames" not in command
    assert "--shift" in command
    assert Path(command[command.index("--output") + 1]).suffix == ".png"


def test_build_minimax_h3_preview_rejects_non_native_video_length(tmp_path: Path):
    paths = []
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(str(path))
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "minimax_h3_dit_model": paths[0],
        "vae_model": paths[1],
        "minimax_h3_text_encoder": paths[2],
        "output_dir": str(tmp_path),
    }

    with pytest.raises(ValueError, match="5, 22, 39"):
        build_minimax_h3_preview(settings, [{"prompt": "one", "frames": 12}])
