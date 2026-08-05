from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sample_gallery import parse_training_sample_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v"}
SAMPLE_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


def discover_samples(output_dir: str, output_name: str = "", limit: int = 300) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        return {"root": str(root), "groups": [], "ungrouped": [], "count": 0}
    search_root = root / output_name if output_name and (root / output_name).is_dir() else root
    candidates = []
    for path in search_root.rglob("*"):
        try:
            if path.is_file() and path.suffix.lower() in SAMPLE_SUFFIXES:
                candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    candidates.sort(reverse=True)
    candidates = candidates[:limit]
    groups: dict[str, list[dict[str, Any]]] = {}
    ungrouped = []
    for modified, path in candidates:
        parsed = parse_training_sample_path(path)
        item = {
            "name": path.name,
            "path": str(path),
            "url": f"/api/sample-file?path={quote(str(path))}",
            "modified": modified,
            "media_kind": "video" if path.suffix.lower() in VIDEO_SUFFIXES else "image",
        }
        if parsed is None:
            ungrouped.append(item)
            continue
        item.update(
            {
                "prefix": parsed["prefix"],
                "seed": parsed["seed"],
                "sequence_kind": parsed["sequence_kind"],
                "sequence": parsed["sequence"],
                "sequence_label": parsed["sequence_label"],
                "prompt_index": parsed.get("prompt_index"),
            }
        )
        groups.setdefault(parsed["group_key"], []).append(item)
    grouped = []
    for key, items in groups.items():
        items.sort(key=lambda item: (item["sequence_kind"] != "epoch", item["sequence"], item["modified"]))
        grouped.append({"key": str(key), "items": items, "modified": max(item["modified"] for item in items)})
    grouped.sort(key=lambda group: group["modified"], reverse=True)
    return {
        "root": str(search_root),
        "groups": grouped,
        "ungrouped": ungrouped,
        "count": len(candidates),
    }


def allowed_sample_roots(settings: dict[str, Any], history: list[dict[str, Any]]) -> list[Path]:
    roots = []
    values = [settings.get("output_dir")]
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
