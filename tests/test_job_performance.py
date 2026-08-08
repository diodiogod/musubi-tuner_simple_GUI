from datetime import datetime, timedelta

import pytest

import job_performance
from modern_gui.jobs import JobSupervisor


def test_training_speed_parser_normalizes_tqdm_units():
    slow = job_performance.parse_training_speed("steps: 10%|#| 10/100 [00:50<07:30, 5.00s/it]")
    fast = job_performance.parse_training_speed("steps: 10%|#| 10/100 [00:02<00:18, 4.00it/s]")

    assert slow == {"step": 10, "total": 100, "seconds_per_iteration": 5.0}
    assert fast == {"step": 10, "total": 100, "seconds_per_iteration": 0.25}
    assert job_performance.parse_training_speed("Loading weights: 10/100 [00:01, 5.0it/s]") is None


def test_performance_summary_prefers_measured_training_samples():
    job = {
        "duration_seconds": 100,
        "current_step": 20,
        "speed_history": [[1, 2.0, "t1"], [2, 4.0, "t2"], [3, 3.0, "t3"]],
    }

    summary = job_performance.performance_summary(job)

    assert summary["median_seconds_per_iteration"] == 3.0
    assert summary["overall_seconds_per_iteration"] == 5.0
    assert summary["quality"] == "measured"


def test_legacy_job_gets_explicit_whole_job_estimate():
    started = datetime(2026, 1, 1, 12, 0, 0)
    job = {
        "started_at": started.isoformat(),
        "finished_at": (started + timedelta(seconds=60)).isoformat(),
        "current_step": 20,
    }

    summary = job_performance.performance_summary(job)

    assert summary["overall_seconds_per_iteration"] == 3.0
    assert summary["sample_count"] == 0
    assert summary["quality"] == "whole-job estimate"


def test_continuation_metadata_tracks_saved_state_from_the_parent_job(tmp_path):
    output = tmp_path / "runs" / "portrait"
    parent = {
        "id": "parent-1",
        "name": "portrait",
        "started_at": "2026-01-01T00:00:00+00:00",
        "settings": {"output_dir": str(tmp_path / "runs"), "output_name": "portrait"},
        "metrics": {"step": 100, "total_steps": 100, "epoch": 4, "total_epochs": 4},
    }
    settings = {
        "output_dir": str(tmp_path / "runs"),
        "output_name": "portrait-cont",
        "resume_path": str(output / "portrait-000002-state"),
    }

    lineage = job_performance.continuation_metadata(settings, [parent])

    assert lineage["continuation_parent_id"] == "parent-1"
    assert lineage["continuation_prior_steps"] == 50
    assert lineage["continuation_prior_epochs"] == 2


def test_performance_summary_exposes_cumulative_continuation_totals():
    summary = job_performance.performance_summary(
        {
            "metrics": {"step": 20, "total_steps": 60, "epoch": 1, "total_epochs": 3},
            "continuation_prior_steps": 100,
            "continuation_prior_epochs": 4,
        }
    )

    assert summary["total_steps"] == 60
    assert summary["cumulative_step"] == 120
    assert summary["cumulative_total_steps"] == 160
    assert summary["cumulative_total_epochs"] == 7


def test_exact_recovery_is_not_counted_as_a_second_continuation():
    assert job_performance.continuation_metadata(
        {"resume_path": "state", "recovery_mode": True}, []
    ) == {}


def test_exact_recovery_preserves_prefix_of_an_additive_child_chain(tmp_path):
    output = tmp_path / "runs" / "portrait-cont"
    parent = {
        "id": "child-1",
        "name": "portrait-cont",
        "started_at": "2026-01-01T00:00:00+00:00",
        "settings": {"output_dir": str(tmp_path / "runs"), "output_name": "portrait-cont"},
        "metrics": {"step": 1200, "total_steps": 2400, "epoch": 2, "total_epochs": 4},
        "continuation_parent_id": "root-1",
        "continuation_parent_title": "portrait-v1",
        "continuation_prior_steps": 5000,
        "continuation_prior_epochs": 6,
        "continuation_depth": 2,
    }
    settings = {
        "starting_point_mode": "state",
        "recovery_mode": True,
        "resume_exact_position": True,
        "resume_path": str(output / "portrait-cont-000002-state"),
    }

    lineage = job_performance.continuation_metadata(settings, [parent])

    assert lineage["continuation_prior_steps"] == 5000
    assert lineage["continuation_prior_epochs"] == 6
    assert lineage["continuation_depth"] == 2


def test_saved_checkpoint_caps_accounted_steps_after_interruption(tmp_path):
    run_dir = tmp_path / "runs" / "portrait"
    run_dir.mkdir(parents=True)
    (run_dir / "portrait-000002-state").mkdir()
    job = {
        "status": "stopped",
        "finished_at": "2030-01-01T00:00:00+00:00",
        "settings": {
            "output_dir": str(tmp_path / "runs"),
            "output_name": "portrait",
            "max_train_steps": 8145,
            "max_train_epochs": 5,
        },
        "metrics": {"step": 3263, "total_steps": 8145, "epoch": 3, "total_epochs": 5},
    }

    progress = job_performance.cumulative_progress(job)

    assert progress["step"] == 3263
    assert progress["saved_step"] == 3258
    assert progress["accounted_step"] == 3258


def test_state_continuation_uses_explicit_step_checkpoint_name(tmp_path):
    parent = {
        "id": "parent-state",
        "name": "portrait-v1",
        "started_at": "2026-01-01T00:00:00+00:00",
        "settings": {"output_dir": str(tmp_path / "runs"), "output_name": "portrait-v1"},
        "metrics": {"step": 5000, "total_steps": 8000, "epoch": 5, "total_epochs": 8},
    }
    settings = {
        "starting_point_mode": "state",
        "resume_path": str(tmp_path / "runs" / "portrait-v1" / "portrait-v1-step00001629-state"),
    }

    lineage = job_performance.continuation_metadata(settings, [parent])

    assert lineage["continuation_prior_steps"] == 1629


def test_weights_continuation_can_match_a_copied_lora_by_source_name(tmp_path):
    parent = {
        "id": "parent-weights",
        "name": "portrait-v1",
        "started_at": "2026-01-01T00:00:00+00:00",
        "settings": {"output_dir": str(tmp_path / "runs"), "output_name": "portrait-v1"},
        "metrics": {"step": 2400, "total_steps": 2400, "epoch": 3, "total_epochs": 3},
    }
    settings = {
        "starting_point_mode": "weights",
        "network_weights": str(tmp_path / "copied_loras" / "portrait-v1.safetensors"),
        "output_dir": str(tmp_path / "runs"),
        "output_name": "portrait-v2",
    }

    lineage = job_performance.continuation_metadata(settings, [parent])

    assert lineage["continuation_parent_id"] == "parent-weights"
    assert lineage["continuation_prior_steps"] == 2400


def test_weights_continuation_uses_the_selected_source_epoch_not_parent_final(tmp_path):
    parent = {
        "id": "parent-ten-epochs",
        "name": "portrait-v1",
        "started_at": "2026-01-01T00:00:00+00:00",
        "settings": {"output_dir": str(tmp_path / "runs"), "output_name": "portrait-v1"},
        "metrics": {"step": 10000, "total_steps": 10000, "epoch": 10, "total_epochs": 10},
    }
    settings = {
        "starting_point_mode": "weights",
        "network_weights": str(tmp_path / "runs" / "portrait-v1" / "portrait-v1-000008-state" / "model.safetensors"),
    }

    lineage = job_performance.continuation_metadata(settings, [parent])

    assert lineage["continuation_prior_steps"] == 8000
    assert lineage["continuation_prior_epochs"] == 8
    assert lineage["continuation_source_epoch"] == 8


def test_managed_console_log_round_trip_and_path_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(job_performance, "JOB_LOG_DIR", tmp_path / "logs")
    path = job_performance.job_log_path("job:1")
    job_performance.append_job_log(path, "now", "output", "hello\r")

    assert "[now] [output] hello" in job_performance.read_job_log(path)
    with pytest.raises(ValueError, match="outside"):
        job_performance.read_job_log(tmp_path / "elsewhere.log")


def test_modern_supervisor_persists_console_and_speed_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(job_performance, "JOB_LOG_DIR", tmp_path / "logs")
    log_path = job_performance.job_log_path("modern-job")
    supervisor = JobSupervisor()
    supervisor._active = {
        "console_log_path": str(log_path),
        "metrics": {"loss_history": [], "speed_history": []},
    }

    supervisor._append_log("output", "steps: 20%|##| 20/100 [01:40<06:40, 5.00s/it, avr_loss=0.2]")

    assert supervisor._active["metrics"]["seconds_per_iteration"] == 5.0
    assert supervisor._active["metrics"]["speed_history"][0][:2] == [20, 5.0]
    assert "5.00s/it" in job_performance.read_job_log(log_path)
