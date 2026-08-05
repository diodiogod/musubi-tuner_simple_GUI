"""Shared job-log persistence and training-throughput summaries for both GUIs."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parent
JOB_LOG_DIR = ROOT / "job_logs"
_PROGRESS_RE = re.compile(r"^steps:\s*\d{1,3}%.*?\b(\d+)\s*/\s*(\d+)\s*\[", re.IGNORECASE)
_RATE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(s/it|it/s)\b", re.IGNORECASE)


def parse_training_speed(text: str) -> dict[str, float | int] | None:
    """Parse the main named tqdm bar and normalize its rate to seconds/iteration."""

    clean = str(text or "").strip()
    progress = _PROGRESS_RE.search(clean)
    rate = _RATE_RE.search(clean)
    if not progress or not rate:
        return None
    value = float(rate.group(1))
    if not math.isfinite(value) or value <= 0:
        return None
    seconds = value if rate.group(2).lower() == "s/it" else 1.0 / value
    return {"step": int(progress.group(1)), "total": int(progress.group(2)), "seconds_per_iteration": seconds}


def compact_speed_history(points: list[Any], limit: int = 1200) -> list[list[Any]]:
    valid: list[list[Any]] = []
    for point in points or []:
        try:
            step, seconds = int(point[0]), float(point[1])
            observed = str(point[2]) if len(point) > 2 else ""
        except (IndexError, TypeError, ValueError):
            continue
        if step >= 0 and seconds > 0 and math.isfinite(seconds):
            item = [step, round(seconds, 6), observed]
            if valid and valid[-1][0] == step:
                valid[-1] = item
            else:
                valid.append(item)
    while len(valid) > limit:
        valid = valid[::2]
    return valid[-limit:]


def job_log_path(job_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(job_id or "job")).strip("._") or "job"
    JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return JOB_LOG_DIR / f"{safe}.log"


def append_job_log(path: str | Path, timestamp: str, stream: str, message: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = str(message).replace("\r", "")
    with target.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{timestamp}] [{stream}] {clean}\n")


def read_job_log(path: str | Path, max_bytes: int = 2_000_000) -> str:
    target = Path(path).expanduser().resolve()
    root = JOB_LOG_DIR.resolve()
    if target != root and root not in target.parents:
        raise ValueError("Recorded console log is outside the managed job-log folder")
    if not target.is_file():
        raise FileNotFoundError("No persisted console log is available for this job")
    size = target.stat().st_size
    with target.open("rb") as handle:
        if size > max_bytes:
            handle.seek(-max_bytes, 2)
            handle.readline()
        payload = handle.read()
    text = payload.decode("utf-8", errors="replace")
    return ("[Earlier log content omitted]\n" if size > max_bytes else "") + text


def _timestamp(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _job_metrics(job: dict[str, Any]) -> dict[str, Any]:
    return job.get("metrics") if isinstance(job.get("metrics"), dict) else {}


def performance_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Return comparable timing information for legacy and newly recorded jobs."""

    metrics = _job_metrics(job)
    points = compact_speed_history(job.get("speed_history") or metrics.get("speed_history") or [])
    seconds = [float(point[1]) for point in points]
    step = int(metrics.get("step") or job.get("current_step") or 0)
    total = int(metrics.get("total_steps") or job.get("total_steps") or 0)
    duration = job.get("duration_seconds")
    if duration is None:
        started, finished = _timestamp(job.get("started_at")), _timestamp(job.get("finished_at"))
        if started and finished:
            try:
                duration = max(0.0, (finished - started).total_seconds())
            except TypeError:
                duration = None
    overall = float(duration) / step if duration is not None and step > 0 else None
    median = statistics.median(seconds) if seconds else None
    recent = statistics.median(seconds[-20:]) if seconds else None
    return {
        "step": step,
        "total_steps": total,
        "duration_seconds": duration,
        "overall_seconds_per_iteration": overall,
        "median_seconds_per_iteration": median,
        "recent_seconds_per_iteration": recent,
        "minimum_seconds_per_iteration": min(seconds) if seconds else None,
        "maximum_seconds_per_iteration": max(seconds) if seconds else None,
        "sample_count": len(seconds),
        "speed_history": points,
        "quality": "measured" if seconds else "whole-job estimate" if overall is not None else "unavailable",
        "console_log_path": str(job.get("console_log_path") or ""),
    }


def normalize_modern_job_for_classic(job: dict[str, Any], index: int) -> dict[str, Any]:
    settings = job.get("settings") if isinstance(job.get("settings"), dict) else {}
    metrics = _job_metrics(job)
    summary = performance_summary(job)
    return {
        "job_id": job.get("id") or f"web-{index}",
        "kind": job.get("kind") or "training",
        "title": job.get("name") or "Modern GUI job",
        "mode": job.get("mode") or settings.get("training_mode") or "",
        "status": job.get("status") or "unknown",
        "started_at": job.get("started_at") or "",
        "finished_at": job.get("finished_at") or "",
        "duration_seconds": summary["duration_seconds"],
        "output_dir": settings.get("output_dir") or "",
        "output_name": settings.get("output_name") or job.get("name") or "",
        "resume_path": settings.get("resume_path") or "",
        "logging_dir": settings.get("logging_dir") or "",
        "note": "Recorded by the modern GUI.",
        "commands": job.get("commands") or [],
        "peak_vram_gb": metrics.get("peak_vram_gb"),
        "last_loss": metrics.get("loss"),
        "current_step": metrics.get("step") or 0,
        "total_steps": metrics.get("total_steps") or 0,
        "current_epoch": metrics.get("epoch") or 0,
        "total_epochs": metrics.get("total_epochs") or 0,
        "loss_history": metrics.get("loss_history") or [],
        "speed_history": summary["speed_history"],
        "console_log_path": job.get("console_log_path") or "",
        "settings_snapshot": settings,
        "_source": "web",
        "_history_index": index,
    }


def load_modern_history_for_classic(path: str | Path | None = None) -> list[dict[str, Any]]:
    source = Path(path) if path else ROOT / "modern_gui_jobs.json"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    return [normalize_modern_job_for_classic(job, index) for index, job in enumerate(payload) if isinstance(job, dict)]


def enrich_job(job: dict[str, Any]) -> dict[str, Any]:
    result = dict(job)
    result["performance"] = performance_summary(job)
    return result
