import sys
import time

import pytest
import psutil

from modern_gui import jobs


def wait_until_finished(supervisor, timeout=10):
    deadline = time.time() + timeout
    snapshot = supervisor.snapshot()
    while time.time() < deadline:
        snapshot = supervisor.snapshot()
        if snapshot["active"]["status"] in {"completed", "failed", "stopped"}:
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"Job did not finish: {snapshot['active']}")


def test_supervisor_runs_sequential_commands_and_records_bounded_log(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    monkeypatch.setattr(
        jobs,
        "build_command_plan",
        lambda settings: {
            "cache": [[sys.executable, "-c", "print('cache-ready')"]],
            "train": [[sys.executable, "-c", "print('train-ready')"]],
        },
    )
    supervisor = jobs.JobSupervisor()

    started = supervisor.start({"output_name": "web-test", "training_mode": "Krea 2"})
    assert started["status"] == "starting"

    snapshot = wait_until_finished(supervisor)

    assert snapshot["active"]["status"] == "completed"
    output = "\n".join(entry["message"] for entry in snapshot["log"])
    assert "cache-ready" in output
    assert "train-ready" in output
    assert supervisor.history()[0]["name"] == "web-test"


def test_supervisor_persists_automatic_training_note(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    monkeypatch.setattr(
        jobs,
        "build_command_plan",
        lambda settings: {"cache": [], "train": [[sys.executable, "-c", "print('train-ready')"]]},
    )
    supervisor = jobs.JobSupervisor()

    supervisor.start(
        {
            "output_name": "automatic-note-test",
            "training_mode": "Krea 2",
            "auto_training_settings_summary": True,
            "max_train_epochs": "2",
        }
    )
    wait_until_finished(supervisor)

    recorded = supervisor.history()[0]["settings"]["training_comment"]
    assert recorded.startswith("Settings: ")
    assert "2 epochs" in recorded


def test_history_repairs_automatic_note_for_older_run(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    jobs._write_history(
        [{
            "name": "older-run",
            "settings": {
                "training_mode": "Krea 2",
                "output_name": "older-run",
                "auto_training_settings_summary": True,
                "max_train_epochs": "3",
            },
        }]
    )

    recorded = jobs.JobSupervisor().history()[0]["settings"]["training_comment"]
    assert recorded.startswith("Settings: ")
    assert "3 epochs" in recorded


def test_supervisor_forces_utf8_for_musubi_bilingual_output(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    message = "学習開始 · 1024×1024 · LoRA α16"
    monkeypatch.setattr(
        jobs,
        "build_command_plan",
        lambda settings: {"cache": [], "train": [[sys.executable, "-c", f"print({message!r})"]]},
    )
    supervisor = jobs.JobSupervisor()

    supervisor.start({"output_name": "unicode-test", "training_mode": "MiniMax H3 (Experimental)"})
    snapshot = wait_until_finished(supervisor)

    assert snapshot["active"]["status"] == "completed"
    assert message in "\n".join(entry["message"] for entry in snapshot["log"])


def test_supervisor_finishes_and_reaps_helper_that_keeps_stdout_open(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    child_pid_path = tmp_path / "child.pid"
    command = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], "
        "stdout=sys.stdout, stderr=sys.stderr); "
        f"open({str(child_pid_path)!r},'w').write(str(p.pid)); print('trainer-exited')"
    )
    monkeypatch.setattr(
        jobs,
        "build_command_plan",
        lambda settings: {"cache": [], "train": [[sys.executable, "-c", command]]},
    )
    supervisor = jobs.JobSupervisor()

    started_at = time.monotonic()
    supervisor.start({"output_name": "held-pipe-test", "training_mode": "MiniMax H3 (Experimental)"})
    snapshot = wait_until_finished(supervisor, timeout=2)

    assert snapshot["active"]["status"] == "completed"
    assert time.monotonic() - started_at < 2
    assert "trainer-exited" in "\n".join(entry["message"] for entry in snapshot["log"])
    child_pid = int(child_pid_path.read_text())
    deadline = time.time() + 2
    while time.time() < deadline and psutil.pid_exists(child_pid):
        time.sleep(0.02)
    assert not psutil.pid_exists(child_pid)


def test_supervisor_rejects_parallel_active_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    monkeypatch.setattr(
        jobs,
        "build_command_plan",
        lambda settings: {"cache": [], "train": [[sys.executable, "-c", "import time; time.sleep(1)"]]},
    )
    supervisor = jobs.JobSupervisor()
    supervisor.start({"output_name": "first"})

    try:
        supervisor.start({"output_name": "second"})
        raise AssertionError("Expected the second start to fail")
    except RuntimeError as exc:
        assert "already active" in str(exc)
    finally:
        supervisor.stop()
        assert wait_until_finished(supervisor)["active"]["status"] == "stopped"


def test_supervisor_runs_utility_commands_through_shared_history(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    supervisor = jobs.JobSupervisor()

    started = supervisor.start_commands(
        [[sys.executable, "-c", "print('utility-ready')"]],
        name="converted.safetensors",
        mode="Utility",
        kind="conversion",
        settings={"output_dir": str(tmp_path)},
    )

    assert started["kind"] == "conversion"
    snapshot = wait_until_finished(supervisor)
    assert snapshot["active"]["status"] == "completed"
    assert "utility-ready" in "\n".join(item["message"] for item in snapshot["log"])
    assert supervisor.history()[0]["kind"] == "conversion"


def test_sample_test_completion_captures_new_prompt_thumbnail(monkeypatch, tmp_path):
    import prompt_library

    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    monkeypatch.setattr(prompt_library, "default_library_root", lambda: tmp_path / "library")
    output = tmp_path / "sample_test"
    output.mkdir()
    image = output / "result.png"
    supervisor = jobs.JobSupervisor()
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(image)!r}).write_bytes(b'png')",
    ]

    supervisor.start_commands(
        [command],
        name="preview",
        mode="Krea 2",
        kind="sample_test",
        settings={"output_name": "portrait"},
        completion_context={
            "kind": "sample_test",
            "save_path": str(output),
            "existing_outputs": [],
            "prompts": [{"prompt": "portrait", "seed": 42}],
            "mode": "Krea 2",
            "output_name": "portrait",
        },
    )

    snapshot = wait_until_finished(supervisor)
    assert snapshot["active"]["captured_thumbnails"] == 1
    assert snapshot["active"]["sample_outputs"] == [str(image.resolve())]
    assert prompt_library.PromptLibraryStore().prompts[0]["prompt_data"]["prompt"] == "portrait"


def test_supervisor_discovers_new_video_preview_outputs(tmp_path):
    output = tmp_path / "sample_test"
    output.mkdir()
    old = output / "old.mp4"
    old.write_bytes(b"old")
    generated = output / "generated.mp4"
    generated.write_bytes(b"video")

    outputs = jobs.JobSupervisor._find_new_sample_outputs(
        {"save_path": str(output), "existing_outputs": [str(old.resolve())]}
    )

    assert outputs == [str(generated.resolve())]


def test_final_artifact_epoch_rename_matches_classic_toggle(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.safetensors").write_bytes(b"lora")
    (run_dir / "run-state").mkdir()

    messages = jobs.JobSupervisor._rename_final_training_artifacts(
        {
            "output_dir": str(tmp_path),
            "output_name": "run",
            "max_train_epochs": "2",
            "rename_final_artifacts_to_epoch": True,
        }
    )

    assert (run_dir / "run-000002.safetensors").is_file()
    assert (run_dir / "run-000002-state").is_dir()
    assert len(messages) == 2


def test_final_artifact_epoch_rename_can_be_disabled(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.safetensors").write_bytes(b"lora")

    messages = jobs.JobSupervisor._rename_final_training_artifacts(
        {
            "output_dir": str(tmp_path),
            "output_name": "run",
            "max_train_epochs": "2",
            "rename_final_artifacts_to_epoch": False,
        }
    )

    assert messages == []
    assert (run_dir / "run.safetensors").is_file()


def test_supervisor_preflights_typed_face_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    supervisor = jobs.JobSupervisor()
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    input_lora = tmp_path / "identity.safetensors"
    input_lora.write_bytes(b"lora")

    monkeypatch.setattr(
        jobs,
        "validate_face_environment",
        lambda config: (_ for _ in ()).throw(ValueError("face models missing")),
    )
    with pytest.raises(ValueError, match="face models missing"):
        supervisor.start(
            {
                "training_mode": "Krea 2",
                "use_staged_training": True,
                "staged_training_config": [
                    {
                        "label": "face",
                        "type": "face_refinement",
                        "enabled": True,
                        "steps": "10",
                    }
                ],
                "face_refinement_config": {
                    "input_mode": "existing_lora",
                    "input_lora": str(input_lora),
                },
            }
        )


def test_supervisor_runs_standard_stages_with_additive_state_handoff(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    dataset_a = tmp_path / "a.toml"
    dataset_b = tmp_path / "b.toml"
    dataset_a.write_text("[[datasets]]", encoding="utf-8")
    dataset_b.write_text("[[datasets]]", encoding="utf-8")
    seen = []

    def fake_plan(stage_settings):
        seen.append(dict(stage_settings))
        return {"cache": [], "train": [[sys.executable, "-c", "print('stage-ready')"]]}

    monkeypatch.setattr(jobs, "build_command_plan", fake_plan)
    monkeypatch.setattr(jobs, "resolve_standard_state", lambda previous: tmp_path / "complete-state")
    supervisor = jobs.JobSupervisor()
    supervisor.start(
        {
            "training_mode": "Krea 2",
            "output_name": "base",
            "output_dir": str(tmp_path),
            "sample_prompts_data": [{"prompt": "base recipe prompt", "seed": 42}],
            "use_staged_training": True,
            "staged_training_config": [
                {"label": "512", "enabled": True, "dataset_config": str(dataset_a), "epochs": "1"},
                {"label": "1024", "enabled": True, "dataset_config": str(dataset_b), "epochs": "2"},
            ],
        }
    )

    snapshot = wait_until_finished(supervisor)

    assert snapshot["active"]["status"] == "completed"
    assert [item["output_name"] for item in seen] == ["base-512px", "base-1024px"]
    assert seen[0]["resume_path"] == ""
    assert seen[1]["resume_path"] == str(tmp_path / "complete-state")
    assert seen[1]["resume_exact_position"] is False
    history = supervisor.history()[0]
    assert history["settings"]["output_name"] == "base"
    assert history["settings"]["sample_prompts_data"] == [{"prompt": "base recipe prompt", "seed": 42}]
    assert history["final_stage_settings"]["output_name"] == "base-1024px"
    assert history["final_stage_settings"]["dataset_config"] == str(dataset_b)
    assert history["final_stage_settings"]["max_train_epochs"] == "2"
    assert [stage["label"] for stage in history["stage_lineage"]] == ["512px", "1024px"]
    assert history["final_stage_artifacts"]["state"] == str(tmp_path / "complete-state")


def test_supervisor_can_handoff_lora_to_fresh_optimizer_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    dataset = tmp_path / "data.toml"; dataset.write_text("[[datasets]]", encoding="utf-8")
    prior_lora = tmp_path / "prior.safetensors"; prior_lora.write_bytes(b"lora")
    seen = []
    monkeypatch.setattr(jobs, "build_command_plan", lambda settings: (seen.append(dict(settings)) or {"cache": [], "train": [[sys.executable, "-c", "print('ok')"]]}))
    monkeypatch.setattr(jobs, "resolve_stage_lora", lambda settings: prior_lora)
    supervisor = jobs.JobSupervisor()
    supervisor.start({"training_mode":"Krea 2","output_name":"run","output_dir":str(tmp_path),"use_staged_training":True,
        "staged_training_config":[{"label":"base","dataset_config":str(dataset),"epochs":"1"},
        {"label":"polish","dataset_config":str(dataset),"epochs":"1","handoff_mode":"weights","learning_rate":"2e-5"}]})
    snapshot = wait_until_finished(supervisor)
    assert snapshot["active"]["status"] == "completed"
    assert seen[1]["resume_path"] == ""
    assert seen[1]["network_weights"] == str(prior_lora)
    assert seen[1]["learning_rate"] == "2e-5"


def test_supervisor_hands_standard_lora_to_face_and_refined_lora_back_to_standard(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    monkeypatch.setattr(jobs, "validate_face_environment", lambda config: None)
    datasets = []
    for name in ("first.toml", "last.toml"):
        path = tmp_path / name
        path.write_text("[[datasets]]", encoding="utf-8")
        datasets.append(path)
    standard_lora = tmp_path / "standard.safetensors"
    standard_lora.write_bytes(b"standard")
    refined_lora = tmp_path / "refined.safetensors"
    standard_settings = []

    def fake_plan(settings):
        standard_settings.append(dict(settings))
        return {"cache": [], "train": [[sys.executable, "-c", "print('standard')"]]}

    def fake_face(base, stage, index, input_lora):
        assert input_lora == standard_lora
        settings = dict(base, stage_type="face_refinement", max_train_steps="5", max_train_epochs="")
        command = [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(refined_lora)!r}).write_bytes(b'refined')",
        ]
        return settings, command, refined_lora

    monkeypatch.setattr(jobs, "build_command_plan", fake_plan)
    monkeypatch.setattr(jobs, "resolve_stage_lora", lambda settings: standard_lora)
    monkeypatch.setattr(jobs, "resolve_standard_state", lambda settings: tmp_path / "state")
    monkeypatch.setattr(jobs, "prepare_face_stage", fake_face)
    supervisor = jobs.JobSupervisor()
    supervisor.start(
        {
            "training_mode": "Krea 2",
            "output_name": "mixed",
            "output_dir": str(tmp_path),
            "use_staged_training": True,
            "face_refinement_config": {},
            "staged_training_config": [
                {"label": "base", "enabled": True, "dataset_config": str(datasets[0]), "epochs": "1"},
                {"label": "face", "type": "face_refinement", "enabled": True, "steps": "5"},
                {"label": "finish", "enabled": True, "dataset_config": str(datasets[1]), "epochs": "1"},
            ],
        }
    )

    snapshot = wait_until_finished(supervisor)

    assert snapshot["active"]["status"] == "completed"
    assert len(standard_settings) == 2
    assert standard_settings[0]["network_weights"] == ""
    assert standard_settings[1]["network_weights"] == str(refined_lora)
    assert standard_settings[1]["resume_path"] == ""
    assert standard_settings[1]["resume_exact_position"] is False
