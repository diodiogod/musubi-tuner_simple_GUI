"""Shared job-log persistence and training-throughput summaries for both GUIs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parent
JOB_LOG_DIR = ROOT / "job_logs"
_PROGRESS_RE = re.compile(r"^steps:\s*\d{1,3}%.*?\b(\d+)\s*/\s*(\d+)\s*\[", re.IGNORECASE)
_RATE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(s/it|it/s)\b", re.IGNORECASE)
_STATE_EPOCH_RE = re.compile(r"-(\d+)-state$", re.IGNORECASE)
_STATE_STEP_RE = re.compile(r"-step(\d+)-state$", re.IGNORECASE)
_LORA_EPOCH_RE = re.compile(r"-(\d+)$", re.IGNORECASE)
_LORA_STEP_RE = re.compile(r"-step(\d+)$", re.IGNORECASE)


def _number(value: Any) -> int:
    try:
        return max(0, int(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def _saved_step_from_artifacts(job: dict[str, Any]) -> int | None:
    """Find the last checkpoint represented by a finished job's output.

    A tqdm counter can be ahead of the last saved LoRA/state when a run is
    interrupted between checkpoint boundaries.  State folder names provide a
    conservative, reproducible step count for lineage accounting.
    """

    settings = _settings_for_job(job)
    output_dir = str(settings.get("output_dir") or job.get("output_dir") or "").strip()
    output_name = str(settings.get("output_name") or job.get("output_name") or job.get("name") or "").strip()
    if not output_dir or not output_name:
        return None
    run_dir = Path(output_dir).expanduser() / output_name
    try:
        entries = list(run_dir.iterdir())
    except OSError:
        return None
    finished = _timestamp(job.get("finished_at"))
    if finished is not None and finished.tzinfo is None:
        finished = finished.replace(tzinfo=timezone.utc)
    metrics = _job_metrics(job)
    total_steps = _number(metrics.get("total_steps") or job.get("total_steps") or settings.get("max_train_steps"))
    total_epochs = _number(metrics.get("total_epochs") or job.get("total_epochs") or settings.get("max_train_epochs"))
    candidates: list[tuple[float, int]] = []
    for entry in entries:
        try:
            if finished is not None:
                modified = datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc)
                if modified > finished:
                    continue
            modified_value = entry.stat().st_mtime
        except OSError:
            continue
        is_state = entry.is_dir() and entry.name.casefold().endswith("-state")
        is_lora = entry.is_file() and entry.suffix.casefold() == ".safetensors"
        if not is_state and not is_lora:
            continue
        name = entry.stem if is_lora else entry.name
        step_match = _STATE_STEP_RE.search(name) if is_state else _LORA_STEP_RE.search(name)
        epoch_match = _STATE_EPOCH_RE.search(name) if is_state else _LORA_EPOCH_RE.search(name)
        if step_match:
            saved_step = _number(step_match.group(1))
        elif epoch_match and total_steps and total_epochs:
            saved_step = round(total_steps * min(_number(epoch_match.group(1)), total_epochs) / total_epochs)
        elif total_steps and str(job.get("status")) == "completed" and (
            (is_state and name.casefold().endswith("-state")) or (is_lora and name.casefold() == output_name.casefold())
        ):
            saved_step = total_steps
        else:
            continue
        candidates.append((modified_value, saved_step))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def cumulative_progress(job: dict[str, Any]) -> dict[str, int]:
    """Return run-local and inherited progress for a job record.

    ``continuation_prior_*`` is deliberately metadata rather than a replacement
    for the trainer's counters.  A continuation therefore keeps its own
    ``current_step/total_steps`` while the history UI can also show the count
    from the first run in the chain.
    """

    metrics = _job_metrics(job)
    step = _number(metrics.get("step") or job.get("current_step"))
    total = _number(metrics.get("total_steps") or job.get("total_steps"))
    epoch = _number(metrics.get("epoch") or job.get("current_epoch"))
    total_epochs = _number(metrics.get("total_epochs") or job.get("total_epochs"))
    saved_step = _saved_step_from_artifacts(job)
    accounted_step = saved_step if saved_step is not None else step
    saved_epoch = round(total_epochs * saved_step / total) if saved_step and total and total_epochs else 0
    # The live epoch counter can already point at the epoch currently being
    # processed when a run is stopped. Lineage is based on durable checkpoints,
    # so its epoch count must use the same conservative boundary as steps.
    accounted_epoch = saved_epoch if saved_step is not None else epoch
    prior_steps = _number(job.get("continuation_prior_steps"))
    prior_epochs = _number(job.get("continuation_prior_epochs"))
    return {
        "step": step,
        "saved_step": saved_step if saved_step is not None else 0,
        "saved_epoch": saved_epoch,
        "accounted_epoch": accounted_epoch,
        "accounted_step": accounted_step,
        "total_steps": total,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "prior_steps": prior_steps,
        "prior_epochs": prior_epochs,
        "cumulative_step": prior_steps + accounted_step,
        "cumulative_total_steps": prior_steps + total,
        "cumulative_epoch": prior_epochs + accounted_epoch,
        "cumulative_total_epochs": prior_epochs + total_epochs,
    }


def _settings_for_job(job: dict[str, Any]) -> dict[str, Any]:
    settings = job.get("settings")
    if isinstance(settings, dict):
        return settings
    settings = job.get("settings_snapshot")
    return settings if isinstance(settings, dict) else {}


def _normal_path(value: Any) -> str:
    if not value:
        return ""
    try:
        return os.path.normcase(os.path.normpath(str(value))).rstrip("\\/")
    except (TypeError, ValueError):
        return str(value).replace("/", "\\").rstrip("\\/").casefold()


def _output_path(job: dict[str, Any]) -> str:
    settings = _settings_for_job(job)
    output_dir = str(settings.get("output_dir") or job.get("output_dir") or "").strip()
    output_name = str(settings.get("output_name") or job.get("output_name") or job.get("name") or "").strip()
    if not output_dir or not output_name:
        return ""
    return _normal_path(os.path.join(output_dir, output_name))


def _state_epoch(path: str) -> int:
    components = str(path).replace("\\", "/").split("/")
    for component in reversed(components):
        match = _STATE_EPOCH_RE.search(component)
        if match:
            return _number(match.group(1))
    return 0


def _state_step(path: str) -> int:
    components = str(path).replace("\\", "/").split("/")
    for component in reversed(components):
        match = _STATE_STEP_RE.search(component)
        if match:
            return _number(match.group(1))
    return 0


def _lora_checkpoint_epoch(path: str) -> int:
    stem = Path(str(path).replace("\\", "/").split("/")[-1]).stem
    match = _LORA_EPOCH_RE.search(stem)
    return _number(match.group(1)) if match else 0


def _lora_checkpoint_step(path: str) -> int:
    stem = Path(str(path).replace("\\", "/").split("/")[-1]).stem
    match = _LORA_STEP_RE.search(stem)
    return _number(match.group(1)) if match else 0


def _is_under(path: str, parent: str) -> bool:
    if not path or not parent:
        return False
    return path == parent or path.startswith(parent + os.sep) or path.startswith(parent + "\\") or path.startswith(parent + "/")


def _find_lineage_parent(settings: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any] | None:
    resume = _normal_path(settings.get("resume_path"))
    weights = _normal_path(settings.get("network_weights"))
    if not resume and not weights:
        return None
    candidates: list[tuple[int, dict[str, Any]]] = []
    for job in history:
        if not isinstance(job, dict):
            continue
        output = _output_path(job)
        if not output:
            continue
        matched = False
        strength = 0
        if resume:
            # State folders are normally directly below the source output
            # directory (run/run-000002-state), so match both exact and nested
            # paths.  The exact resume path is also useful for older records.
            old_resume = _normal_path(_settings_for_job(job).get("resume_path") or job.get("resume_path"))
            # Retry records that point at this exact state are attempts of the
            # same continuation, not ancestors in the chain.
            if old_resume and old_resume == resume:
                continue
            matched = _is_under(resume, output) or (old_resume and old_resume == resume)
            strength = 3 if _is_under(resume, output) else 2 if matched else 0
        elif weights:
            matched = _is_under(weights, output)
            if matched:
                strength = 3
            else:
                # A LoRA may be copied out of the original run directory. In
                # that case the full path no longer proves ancestry, but the
                # source run name is commonly retained in the filename or one
                # of its parent folders.
                source_name = Path(output).name.casefold()
                strength = 1 if source_name and source_name in weights.casefold() else 0
                matched = strength > 0
        if matched:
            progress = cumulative_progress(job)
            has_progress = int(progress["cumulative_step"] > 0)
            candidates.append((strength, has_progress, job))
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2].get("started_at") or "")), reverse=True)
    return candidates[0][2] if candidates else None


def continuation_metadata(settings: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Infer inherited counters for an additive continuation.

    Exact positional recovery is the same logical run and must not add a second
    copy of its target total.  Ordinary state/weights continuations get a
    recorded prior count so the next history entry can report chain totals.
    """

    exact_recovery = bool(settings.get("recovery_mode") or settings.get("resume_exact_position"))
    if exact_recovery:
        # Exact recovery is not a new additive run, but a recovered attempt of
        # the same logical run. If that logical run was itself a continuation,
        # preserve its inherited prefix without adding the parent's local
        # budget a second time.
        resume = str(settings.get("resume_path") or "").strip()
        parent = _find_lineage_parent(settings, history or []) if resume else None
        prior_steps = _number(parent.get("continuation_prior_steps")) if parent else 0
        prior_epochs = _number(parent.get("continuation_prior_epochs")) if parent else 0
        if not parent or not prior_steps:
            return {}
        return {
            "continuation_parent_id": parent.get("id") or parent.get("job_id") or "",
            "continuation_parent_title": parent.get("name") or parent.get("output_name") or parent.get("title") or "",
            "continuation_prior_steps": prior_steps,
            "continuation_prior_epochs": prior_epochs,
            "continuation_depth": _number(parent.get("continuation_depth")) or 1,
            "continuation_resume_state": resume,
            "continuation_source_checkpoint": resume,
            "continuation_source_step": _state_step(resume),
            "continuation_source_epoch": _state_epoch(resume),
        }
    # A new run may still carry a model/LoRA field in an older saved recipe;
    # only the explicit continuation choices establish lineage.
    if settings.get("starting_point_mode") == "new":
        return {}
    resume = str(settings.get("resume_path") or "").strip()
    weights = str(settings.get("network_weights") or "").strip()
    if not resume and not weights:
        return {}
    history = history or []
    parent = _find_lineage_parent(settings, history)
    source_path = resume or weights
    checkpoint_step = _state_step(source_path) or _lora_checkpoint_step(source_path)
    checkpoint_epoch = _state_epoch(source_path) or _lora_checkpoint_epoch(source_path)
    if parent:
        parent_progress = cumulative_progress(parent)
        prior_steps = parent_progress["cumulative_step"]
        prior_epochs = parent_progress["cumulative_epoch"]
        parent_total_epochs = parent_progress["total_epochs"]
        parent_total_steps = parent_progress["total_steps"]
        if checkpoint_step:
            prior_steps = parent_progress["prior_steps"] + checkpoint_step
            prior_epochs = parent_progress["prior_epochs"]
            if parent_total_epochs and parent_total_steps:
                prior_epochs += round(parent_total_epochs * min(checkpoint_step, parent_total_steps) / parent_total_steps)
        elif checkpoint_epoch and parent_total_epochs and parent_total_steps:
            local_step = round(parent_total_steps * min(checkpoint_epoch, parent_total_epochs) / parent_total_epochs)
            prior_steps = parent_progress["prior_steps"] + local_step
            prior_epochs = parent_progress["prior_epochs"] + min(checkpoint_epoch, parent_total_epochs)
        return {
            "continuation_parent_id": parent.get("id") or parent.get("job_id") or "",
            "continuation_parent_title": parent.get("name") or parent.get("output_name") or parent.get("title") or "",
            "continuation_prior_steps": prior_steps,
            "continuation_prior_epochs": prior_epochs,
            "continuation_depth": _number(parent.get("continuation_depth")) + 1,
            "continuation_resume_state": resume,
            "continuation_source_checkpoint": source_path,
            "continuation_source_step": checkpoint_step,
            "continuation_source_epoch": checkpoint_epoch,
        }
    epoch = checkpoint_epoch
    return {
        "continuation_parent_id": "",
        "continuation_parent_title": Path(resume).parent.name if resume else Path(weights).name,
        "continuation_prior_steps": 0,
        "continuation_prior_epochs": epoch,
        "continuation_depth": 1,
        "continuation_resume_state": resume,
        "continuation_source_checkpoint": source_path,
        "continuation_source_step": checkpoint_step,
        "continuation_source_epoch": checkpoint_epoch,
    }


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
    progress = cumulative_progress(job)
    points = compact_speed_history(job.get("speed_history") or metrics.get("speed_history") or [])
    seconds = [float(point[1]) for point in points]
    step = progress["step"]
    total = progress["total_steps"]
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
        "saved_step": progress["saved_step"],
        "saved_epoch": progress["saved_epoch"],
        "accounted_epoch": progress["accounted_epoch"],
        "accounted_step": progress["accounted_step"],
        "prior_steps": progress["prior_steps"],
        "prior_epochs": progress["prior_epochs"],
        "cumulative_step": progress["cumulative_step"],
        "cumulative_total_steps": progress["cumulative_total_steps"],
        "cumulative_epoch": progress["cumulative_epoch"],
        "cumulative_total_epochs": progress["cumulative_total_epochs"],
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
        "continuation_parent_id": job.get("continuation_parent_id") or "",
        "continuation_parent_title": job.get("continuation_parent_title") or "",
        "continuation_prior_epochs": job.get("continuation_prior_epochs") or 0,
        "continuation_prior_steps": job.get("continuation_prior_steps") or 0,
        "continuation_depth": job.get("continuation_depth") or 0,
        "continuation_resume_state": job.get("continuation_resume_state") or "",
        "continuation_source_checkpoint": job.get("continuation_source_checkpoint") or "",
        "continuation_source_step": job.get("continuation_source_step") or 0,
        "continuation_source_epoch": job.get("continuation_source_epoch") or 0,
        "loss_history": metrics.get("loss_history") or [],
        "speed_history": summary["speed_history"],
        "saved_step": summary["saved_step"],
        "saved_epoch": summary["saved_epoch"],
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
