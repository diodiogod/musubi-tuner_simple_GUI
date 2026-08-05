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
