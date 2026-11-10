"""Read-only run charts/artifacts and local annotations, independent of recipes."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
import threading

from job_performance import read_job_log
from modern_gui.monitor import parse_training_line
from modern_gui.samples import discover_samples

NOTES_PATH = Path(__file__).resolve().parents[1] / "job_notes_local.json"
_lock = threading.RLock()
_cache: OrderedDict = OrderedDict()


def run_key(job: dict) -> str:
    """Stable across sorting and insertion, including classic entries without IDs."""
    identity = [job.get("_source", "desktop"), job.get("id") or job.get("job_id")]
    if not identity[1]:
        settings = job.get("settings_snapshot") or job.get("settings") or {}
        identity += [job.get("started_at"), job.get("name") or job.get("title"),
                     settings.get("output_dir") or job.get("output_dir"),
                     settings.get("output_name") or job.get("output_name"), job.get("commands")]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]


def load_notes(*, strict: bool = False) -> dict:
    with _lock:
        try:
            value = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Invalid run-notes document")
            return value
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            if strict:
                raise ValueError("The local run-notes file is unreadable; it was left unchanged.") from exc
            return {}


def save_note(job: dict, key: str, text: str) -> dict:
    if key != run_key(job):
        raise ValueError("History changed. Refresh the run before saving its note.")
    if not isinstance(text, str) or len(text) > 20000:
        raise ValueError("Run notes must be text with at most 20,000 characters.")
    with _lock:
        notes = load_notes(strict=True)
        notes[key] = text
        temporary = NOTES_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(NOTES_PATH)
    return {"key": key, "note": text}


def decorate_run(job: dict, notes: dict) -> dict:
    return dict(job, insights_key=run_key(job), user_note=notes.get(run_key(job), ""))


def record_chart_point(metrics: dict, update: dict, *, bounded: bool = True) -> None:
    """Keep the old loss_history contract; richer points carry optional raw loss/LR."""
    if "loss" not in update:
        return
    point = [int(metrics.get("step") or 0), metrics.get("epoch"),
             update.get("raw_loss"), update["loss"], update.get("learning_rate")]
    points = metrics.setdefault("chart_history", [])
    if points and point[0] < points[-1][0]:
        points.clear()  # A new stage can restart its local step numbering.
    if points and points[-1][0] == point[0]:
        points[-1] = point
    else:
        points.append(point)
    if bounded and len(points) > 1600:
        metrics["chart_history"] = points[::2] + [points[-1]]


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def chart_payload(metrics: dict) -> dict:
    points = metrics.get("chart_history") or [
        [p[0], None, None, p[1], None] for p in metrics.get("loss_history", [])
        if isinstance(p, (list, tuple)) and len(p) >= 2
    ]
    points = [list(p) for p in points if len(p) == 5 and _finite(p[0]) and _finite(p[3])]
    return {"points": points, "has_raw": any(_finite(p[2]) for p in points),
            "has_lr": any(_finite(p[4]) for p in points),
            "note": "Epoch means use captured updates. Missing steps are not estimated. Lower loss does not guarantee better samples."}


def run_chart(job: dict) -> dict:
    metrics = job.get("metrics") or job
    path = str(job.get("console_log_path") or "")
    try:
        stat = Path(path).stat()
        signature = (path, stat.st_mtime_ns, stat.st_size)
        with _lock:
            cached = _cache.get(signature)
            if cached is not None:
                _cache.move_to_end(signature)
                return cached
        text = read_job_log(path, max_bytes=8_000_000)
    except (OSError, ValueError):
        return chart_payload(metrics)
    recovered: dict = {}
    for line in text.splitlines():
        # Managed logs prefix messages with [timestamp] [stream].
        clean = re.sub(r"^\[[^\]]+\]\s+\[[^\]]+\]\s*", "", line)
        update = parse_training_line(clean)
        recovered.update({k: v for k, v in update.items() if k != "loss"})
        record_chart_point(recovered, update, bounded=False)
    payload = chart_payload(recovered) if recovered.get("chart_history") else chart_payload(metrics)
    if text.startswith("[Earlier log content omitted]"):
        payload["note"] += " Only the most recent 8 MB of the saved log is available in this chart."
    with _lock:
        _cache[signature] = payload
        while len(_cache) > 48:
            _cache.popitem(last=False)
    return payload


def run_files(job: dict) -> dict:
    settings = job.get("settings_snapshot") or job.get("settings") or {}
    output = settings.get("output_dir") or job.get("output_dir")
    name = str(settings.get("output_name") or job.get("output_name") or job.get("name") or "")
    if not output:
        return {"checkpoints": [], "samples": [], "root": ""}
    root = Path(str(output)).expanduser().resolve()
    # Avoid scanning every other run when the recorded output has disappeared.
    if name and (root / name).is_dir():
        root = (root / name).resolve()
    if not root.is_dir():
        return {"checkpoints": [], "samples": [], "root": str(root)}
    def belongs(filename):
        return not name or filename == name or filename.startswith(tuple(name + delimiter for delimiter in (".", "-", "_")))
    checkpoints = []
    for candidate in root.iterdir():
        if not belongs(candidate.name):
            continue
        if candidate.is_file() and candidate.suffix.lower() == ".safetensors":
            stat = candidate.stat()
            checkpoints.append({"name": candidate.name, "size": stat.st_size, "modified": stat.st_mtime, "kind": "LoRA"})
        elif candidate.is_dir() and candidate.name.endswith("-state"):
            checkpoints.append({"name": candidate.name, "size": None, "modified": candidate.stat().st_mtime, "kind": "Resume state"})
    # Restrict flat legacy folders to this run's file prefix as well.
    gallery = discover_samples(str(root), "", limit=240, filename_prefix=name)
    samples = gallery["ungrouped"] + [item for group in gallery["groups"] for item in group["items"]]
    samples = [p for p in samples if belongs(p["name"])]
    return {"root": str(root), "checkpoints": sorted(checkpoints, key=lambda p: p["modified"], reverse=True),
            "samples": sorted(samples, key=lambda p: p["modified"], reverse=True),
            "scope_note": "Files currently in the recorded output folder, including earlier attempts that reused it. Up to 240 recent previews are shown."}
