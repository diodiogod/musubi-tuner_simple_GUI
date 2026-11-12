"""Small cached headers and saved-run labels; never decode model weights/media."""
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATHS = (ROOT / "modern_gui_jobs.json", ROOT / "job_history_local.json")


@lru_cache(maxsize=16)
def _run_records(signatures):
    records = []
    for path, *_ in signatures:
        try:
            jobs = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if not isinstance(job, dict) or job.get("kind", "training") not in {"training", "staged_training"}:
                continue
            settings = job.get("settings") or job.get("settings_snapshot") or {}
            if not isinstance(settings, dict):
                continue
            prompts = [p for p in (settings.get("sample_prompts_data") or []) if isinstance(p, dict) and p.get("enabled", True)]
            output = settings.get("output_dir") or job.get("output_dir")
            name = settings.get("output_name") or job.get("output_name")
            if not prompts or not output or not name:
                continue
            try:
                started = datetime.fromisoformat(str(job.get("started_at")).replace("Z", "+00:00")).timestamp()
                records.append((str(name).casefold(), Path(output).expanduser().resolve(), started, prompts))
            except (OSError, ValueError, TypeError):
                continue
    return records


@lru_cache(maxsize=1200)
def _image_header(path, modified, size):
    try:
        from PIL import Image
        with Image.open(path) as image:
            text = image.info.get("prompt") or image.info.get("positive_prompt") or ""
            # ComfyUI's "prompt" can be a workflow JSON, not a natural-language caption.
            text = text if isinstance(text, str) and not text.lstrip().startswith(("{", "[")) else ""
            return {"width": image.width, "height": image.height, "prompt_text": text[:16000]}
    except (ImportError, OSError, ValueError):
        return {}


def describe_sample(path: Path, parsed: dict | None) -> dict:
    try:
        stat = path.stat()
    except OSError:
        return {}
    result = dict(_image_header(str(path), stat.st_mtime_ns, stat.st_size)) if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else {}
    if result.get("prompt_text"):
        result["metadata_source"] = "Image metadata"
    if parsed:
        signatures = []
        for history in HISTORY_PATHS:
            try:
                state = history.stat()
                signatures.append((str(history), state.st_mtime_ns, state.st_size))
            except OSError:
                pass
        try:
            generated = datetime.strptime(parsed.get("generated_at", ""), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            generated = stat.st_mtime
        matches = [record for record in _run_records(tuple(signatures))
                   if record[0] == parsed["prefix"].casefold() and record[1] in path.resolve().parents and record[2] <= generated + 1]
        if matches:
            record = max(matches, key=lambda r: r[2])
            index = parsed.get("prompt_index", -1)
            if 0 <= index < len(record[3]):
                prompt = record[3][index]
                if not result.get("prompt_text"):
                    result["prompt_text"] = str(prompt.get("prompt") or "")[:16000]
                    result["metadata_source"] = "Saved run settings"
                result["prompt_name"] = str(prompt.get("name") or prompt.get("_library_name") or "")[:160]
                result["sample_settings"] = {key: prompt[key] for key in ("steps", "guidance", "sampler", "scheduler", "frames", "neg") if key in prompt}
    if result.get("prompt_text"):
        result["prompt_identity"] = hashlib.sha256(" ".join(result["prompt_text"].split()).encode()).hexdigest()[:20]
    return result
