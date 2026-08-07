from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sample_gallery import parse_training_sample_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v"}
SAMPLE_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES
SOURCES_PATH = Path(__file__).resolve().parents[1] / "modern_gui_sample_sources.json"


def _resolved_directory(value: str | Path) -> Path:
    return Path(str(value)).expanduser().resolve()


def load_sample_sources() -> list[dict[str, str]]:
    """Load user-added sample roots without making them part of the training recipe."""

    try:
        payload = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    values = payload.get("sources", payload) if isinstance(payload, (dict, list)) else []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, str):
            path, label = item, ""
        elif isinstance(item, dict):
            path, label = str(item.get("path") or ""), str(item.get("label") or "")
        else:
            continue
        if not path:
            continue
        try:
            resolved = _resolved_directory(path)
        except (OSError, ValueError):
            continue
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append({"path": str(resolved), "label": label.strip() or resolved.name})
    return result


def _save_sample_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    temporary = SOURCES_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"sources": sources}, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(SOURCES_PATH)
    return sources


def add_sample_source(path: str, label: str = "") -> list[dict[str, str]]:
    resolved = _resolved_directory(path)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Sample folder does not exist: {resolved}")
    if not any(candidate.is_file() and candidate.suffix.lower() in SAMPLE_SUFFIXES for candidate in resolved.rglob("*")):
        raise ValueError("The selected folder does not contain supported sample images or videos.")
    sources = load_sample_sources()
    key = str(resolved).casefold()
    sources = [item for item in sources if item["path"].casefold() != key]
    sources.append({"path": str(resolved), "label": label.strip() or resolved.name})
    return _save_sample_sources(sources)


def remove_sample_source(path: str) -> list[dict[str, str]]:
    resolved = str(_resolved_directory(path)).casefold()
    return _save_sample_sources([item for item in load_sample_sources() if item["path"].casefold() != resolved])


def sample_source_status() -> list[dict[str, Any]]:
    result = []
    for item in load_sample_sources():
        path = Path(item["path"])
        try:
            count = sum(1 for candidate in path.rglob("*") if candidate.is_file() and candidate.suffix.lower() in SAMPLE_SUFFIXES)
        except OSError:
            count = 0
        result.append({**item, "exists": path.is_dir(), "count": count})
    return result


def find_nearby_sample_sources(output_dir: str, output_name: str = "") -> list[dict[str, Any]]:
    """Find sibling output folders containing samples, without adding them."""

    root = _resolved_directory(output_dir)
    current = root / output_name if output_name and (root / output_name).is_dir() else root
    parent = current.parent if current != root else root
    try:
        candidates = sorted(parent.iterdir(), key=lambda item: item.name.casefold())
    except OSError:
        return []
    result = []
    current_resolved = current.resolve()
    for candidate in candidates:
        try:
            if not candidate.is_dir() or candidate.resolve() == current_resolved:
                continue
            count = sum(1 for item in candidate.rglob("*") if item.is_file() and item.suffix.lower() in SAMPLE_SUFFIXES)
        except OSError:
            continue
        if count:
            result.append({"path": str(candidate.resolve()), "label": candidate.name, "count": count})
    return result


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (ImportError, OSError, ValueError):
        return None


def discover_samples(
    output_dir: str,
    output_name: str = "",
    limit: int = 300,
    source_paths: list[dict[str, str] | str] | None = None,
) -> dict[str, Any]:
    root = _resolved_directory(output_dir)
    search_root = root / output_name if output_name and (root / output_name).is_dir() else root
    roots: list[tuple[Path, str]] = []
    if search_root.is_dir():
        roots.append((search_root, output_name.strip() or search_root.name or "Current run"))
    for value in source_paths or []:
        if isinstance(value, dict):
            candidate, label = value.get("path", ""), value.get("label", "")
        else:
            candidate, label = value, ""
        try:
            source_root = _resolved_directory(candidate)
        except (OSError, ValueError):
            continue
        if not source_root.is_dir() or any(source_root == existing for existing, _ in roots):
            continue
        roots.append((source_root, str(label).strip() or source_root.name or "Previous run"))
    candidates = []
    for source_index, (source_root, source_label) in enumerate(roots):
        for path in source_root.rglob("*"):
            try:
                if path.is_file() and path.suffix.lower() in SAMPLE_SUFFIXES:
                    candidates.append((path.stat().st_mtime, source_index, source_label, path))
            except OSError:
                continue
    candidates.sort(reverse=True)
    candidates = candidates[:limit]
    groups: dict[str, list[dict[str, Any]]] = {}
    ungrouped = []
    multi_source = len(roots) > 1
    for modified, source_index, source_label, path in candidates:
        parsed = parse_training_sample_path(path)
        item = {
            "name": path.name,
            "path": str(path),
            "url": f"/api/sample-file?path={quote(str(path))}",
            "modified": modified,
            "media_kind": "video" if path.suffix.lower() in VIDEO_SUFFIXES else "image",
            "source_index": source_index,
            "source_label": source_label,
        }
        if parsed is None:
            ungrouped.append(item)
            continue
        dimensions = _image_dimensions(path)
        item.update(
            {
                "prefix": parsed["prefix"],
                "seed": parsed["seed"],
                "sequence_kind": parsed["sequence_kind"],
                "sequence": parsed["sequence"],
                "sequence_label": parsed["sequence_label"],
                "prompt_index": parsed.get("prompt_index"),
                "width": dimensions[0] if dimensions else None,
                "height": dimensions[1] if dimensions else None,
            }
        )
        if multi_source:
            dimension_key = f"{dimensions[0]}x{dimensions[1]}" if dimensions else "unknown"
            key = "|".join(
                (
                    f"prompt:{parsed.get('prompt_index')}",
                    f"seed:{parsed.get('seed') or ''}",
                    f"tail:{parsed.get('tail') or ''}",
                    f"kind:{parsed['sequence_kind']}",
                    f"format:{path.suffix.casefold()}",
                    f"size:{dimension_key}",
                )
            )
        else:
            key = parsed["group_key"]
        groups.setdefault(key, []).append(item)
    grouped = []
    for key, items in groups.items():
        items.sort(key=lambda item: (item["sequence_kind"] != "epoch", item["sequence"], item["source_index"], item["modified"]))
        grouped.append(
            {
                "key": str(key),
                "items": items,
                "modified": max(item["modified"] for item in items),
                "source_labels": list(dict.fromkeys(item["source_label"] for item in items)),
                "prompt_index": items[0].get("prompt_index"),
                "seed": items[0].get("seed"),
            }
        )
    grouped.sort(key=lambda group: group["modified"], reverse=True)
    return {
        "root": str(search_root),
        "sources": [{"path": str(path), "label": label} for path, label in roots],
        "groups": grouped,
        "ungrouped": ungrouped,
        "count": len(candidates),
    }


def allowed_sample_roots(settings: dict[str, Any], history: list[dict[str, Any]]) -> list[Path]:
    roots = []
    values = [settings.get("output_dir")]
    values.extend(item["path"] for item in load_sample_sources())
    for job in history:
        snapshot = job.get("settings_snapshot") or job.get("settings") or {}
        values.extend((job.get("output_dir"), snapshot.get("output_dir")))
    for value in values:
        if not value:
            continue
        try:
            path = Path(str(value)).expanduser().resolve()
        except (OSError, ValueError):
            continue
        if path not in roots:
            roots.append(path)
    return roots


def resolve_sample_file(path: str, allowed_roots: list[Path]) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() not in SAMPLE_SUFFIXES or not resolved.is_file():
        raise FileNotFoundError("Sample media does not exist or has an unsupported format.")
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise PermissionError("Sample path is outside configured output directories.")
    return resolved, mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
