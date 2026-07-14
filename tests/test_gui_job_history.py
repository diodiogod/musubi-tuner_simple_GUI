from musubi_tuner_gui import MusubiTunerGUI


def test_job_display_name_prefers_recorded_output_name():
    job = {"output_name": "character_v4", "title": "Training run"}
    assert MusubiTunerGUI._job_display_name(job) == "character_v4"


def test_job_display_name_supports_older_settings_snapshot():
    job = {
        "output_name": "",
        "settings_snapshot": {"output_name": "older_run"},
        "title": "Krea Turbo face evaluation",
    }
    assert MusubiTunerGUI._job_display_name(job) == "older_run"


def test_job_display_name_has_readable_fallback():
    job = {"output_dir": "C:/models/example_run", "title": "Training run"}
    assert MusubiTunerGUI._job_display_name(job) == "example_run"


def test_complete_accelerate_state_is_a_true_resume(tmp_path):
    state = tmp_path / "run-000016-state"
    state.mkdir()
    for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
        (state / name).write_bytes(b"state")

    valid, missing = MusubiTunerGUI._validate_accelerate_state(state)

    assert valid is True
    assert missing == []


def test_incomplete_accelerate_state_is_not_called_a_true_resume(tmp_path):
    state = tmp_path / "run-000016-state"
    state.mkdir()
    (state / "model.safetensors").write_bytes(b"state")

    valid, missing = MusubiTunerGUI._validate_accelerate_state(state)

    assert valid is False
    assert "optimizer state" in missing
    assert "scheduler state" in missing
    assert "random-number state" in missing


def test_unnumbered_state_is_not_presented_as_exact_position(tmp_path):
    state = tmp_path / "run-state"
    state.mkdir()
    for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
        (state / name).write_bytes(b"state")

    valid, missing = MusubiTunerGUI._validate_accelerate_state(state)

    assert valid is False
    assert "epoch/step position marker in the state-folder name" in missing


def test_loss_history_is_bounded_and_keeps_endpoints():
    history = [(step, step / 1000) for step in range(5000)]

    compact = MusubiTunerGUI._compact_loss_history(history, limit=100)

    assert len(compact) <= 100
    assert compact[0] == [0, 0.0]
    assert compact[-1] == [4999, 4.999]
