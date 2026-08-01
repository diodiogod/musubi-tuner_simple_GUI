from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from modern_gui.dataset_documents import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    plain_document,
    parse_document,
    resolve_configured_path,
)


MAX_PAGE_SIZE = 48
MAX_CAPTION_BYTES = 256 * 1024
TOKEN_TTL_SECONDS = 30 * 60
MAX_REGISTERED_ITEMS = 4096


class MediaTokenError(PermissionError):
    pass


class CaptionConflictError(RuntimeError):
    pass


@dataclass
class MediaAccess:
    path: Path
    kind: str
    source_root: Path | None
    caption_mode: str
    caption_path: Path | None = None
    manifest_path: Path | None = None
    manifest_line: int | None = None
    media_key: str = ""
    media_value: str = ""
    allow_descendant: bool = False
    issued_at: float = 0.0


_TOKENS: dict[str, MediaAccess] = {}
_TOKEN_LOCK = threading.Lock()


def _file_revision(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cleanup_tokens(now: float) -> None:
    expired = [
        token
        for token, access in _TOKENS.items()
        if now - access.issued_at > TOKEN_TTL_SECONDS
    ]
    for token in expired:
        _TOKENS.pop(token, None)
    if len(_TOKENS) > MAX_REGISTERED_ITEMS:
        oldest = sorted(_TOKENS, key=lambda key: _TOKENS[key].issued_at)
        for token in oldest[: len(_TOKENS) - MAX_REGISTERED_ITEMS]:
            _TOKENS.pop(token, None)


def _register(access: MediaAccess) -> str:
    now = time.monotonic()
    token = secrets.token_urlsafe(24)
    access.issued_at = now
    with _TOKEN_LOCK:
        _cleanup_tokens(now)
        _TOKENS[token] = access
    return token


def resolve_media_token(token: str) -> MediaAccess:
    now = time.monotonic()
    with _TOKEN_LOCK:
        _cleanup_tokens(now)
        access = _TOKENS.get(str(token or ""))
    if access is None:
        raise MediaTokenError("This media preview expired. Refresh the dataset browser.")
    path = access.path.resolve()
    if access.source_root is not None:
        root = access.source_root.resolve()
        contained = root in path.parents if access.allow_descendant else path.parent == root
        if not contained:
            raise MediaTokenError("The media file is no longer inside its configured dataset source.")
    if not path.is_file():
        raise FileNotFoundError("The media file no longer exists.")
    access.path = path
    return access


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        (0, int(piece)) if piece.isdigit() else (1, piece.casefold())
        for piece in re.split(r"(\d+)", path.name)
    )


def _normalize_caption_extension(value: Any) -> str:
    extension = str(value or "").strip()
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    if extension and ("/" in extension or "\\" in extension or extension in {".", ".."}):
        raise ValueError("Caption extension must be a filename extension such as .txt.")
    return extension


def _read_caption(path: Path | None) -> tuple[str, str, str]:
    if path is None:
        return "", "not_configured", "missing"
    if not path.is_file():
        return "", "missing", "missing"
    revision = _file_revision(path)
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_CAPTION_BYTES:
            return "", "too_large", revision
        caption = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "", "unreadable", revision
    return caption, "empty" if not caption.strip() else "present", revision


def _image_dimensions(path: Path) -> tuple[int | None, int | None, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            orientation = image.getexif().get(274)
            if orientation in {5, 6, 7, 8}:
                width, height = height, width
        return width, height, ""
    except (OSError, ValueError) as exc:
        return None, None, str(exc)


def _source_spec(text: str, index: int) -> dict[str, Any]:
    plain = plain_document(parse_document(text))
    datasets = plain.get("datasets", [])
    if not isinstance(datasets, list) or index < 0 or index >= len(datasets):
        raise IndexError(f"Dataset index {index} does not exist.")
    dataset = datasets[index]
    if not isinstance(dataset, dict):
        raise ValueError("The selected dataset is not a TOML table.")
    general = plain.get("general", {})
    if not isinstance(general, dict):
        general = {}
    kind = "video" if "video_directory" in dataset or "video_jsonl_file" in dataset else "image"
    directory_key = f"{kind}_directory"
    jsonl_key = f"{kind}_jsonl_file"
    directory = str(dataset.get(directory_key) or "").strip()
    manifest = str(dataset.get(jsonl_key) or "").strip()
    if bool(directory) == bool(manifest):
        raise ValueError(f"Source {index + 1} must use exactly one {kind} directory or JSONL file.")
    mode = "directory" if directory else "jsonl"
    source_value = directory or manifest
    caption_extension = _normalize_caption_extension(
        dataset.get("caption_extension", general.get("caption_extension"))
    )
    repeats = dataset.get("num_repeats", general.get("num_repeats", 1))
    try:
        repeats = max(1, int(repeats))
    except (TypeError, ValueError):
        repeats = 1
    return {
        "index": index,
        "kind": kind,
        "mode": mode,
        "source": source_value,
        "path": resolve_configured_path(source_value),
        "dataset": dataset,
        "general": general,
        "caption_extension": caption_extension,
        "repeats": repeats,
    }


def _control_candidates(root: Path | None, media: Path, kind: str) -> list[Path]:
    if root is None or not root.is_dir():
        return []
    resolved_root = root.resolve()
    if kind == "video":
        direct = root / media.name
        stem_directory = root / media.stem
        candidates = [direct.resolve()] if direct.is_file() and direct.resolve().parent == resolved_root else []
        if stem_directory.is_dir() and stem_directory.resolve().parent == resolved_root:
            frames = sorted(
                (
                    path.resolve()
                    for path in stem_directory.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                    and resolved_root in path.resolve().parents
                ),
                key=_natural_key,
            )
            return frames[:1]
        candidates.extend(
            path.resolve()
            for path in root.iterdir()
            if path.is_file()
            and path.stem.casefold() == media.stem.casefold()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and path.resolve().parent == resolved_root
            and path.resolve() != direct.resolve()
        )
        return candidates[:4]
    prefix = f"{media.stem.casefold()}_"
    candidates = [
        path.resolve()
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.resolve().parent == resolved_root
        and (path.stem.casefold() == media.stem.casefold() or path.stem.casefold().startswith(prefix))
    ]
    return sorted(candidates, key=_natural_key)[:4]


def _directory_control_map(
    root: Path | None,
    media_paths: list[Path],
    kind: str,
) -> dict[Path, list[Path]]:
    if root is None or not root.is_dir():
        return {}
    if kind == "video":
        return {
            path: matches
            for path in media_paths
            if (matches := _control_candidates(root, path, kind))
        }
    controls = {
        path.resolve()
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.resolve().parent == root.resolve()
    }
    paired: dict[Path, list[Path]] = {}
    # Musubi considers longer media basenames first, then consumes matches so
    # overlapping stems cannot claim the same control twice.
    for media in sorted(media_paths, key=lambda path: len(path.name), reverse=True):
        stem = media.stem.casefold()
        candidates = [
            path
            for path in controls
            if path.stem.casefold() == stem or path.stem.casefold().startswith(f"{stem}_")
        ]
        candidates.sort(
            key=lambda path: (
                0
                if path.stem.casefold() == stem
                else (
                    int(path.stem.rsplit("_", 1)[-1]) + 1
                    if path.stem.rsplit("_", 1)[-1].isdigit()
                    else 10**9
                )
            )
        )
        if candidates:
            paired[media] = candidates[:4]
            controls.difference_update(candidates)
    return paired


def _register_preview(
    path: Path,
    *,
    kind: str,
    source_root: Path | None,
    caption_mode: str = "none",
    caption_path: Path | None = None,
    manifest_path: Path | None = None,
    manifest_line: int | None = None,
    media_key: str = "",
    media_value: str = "",
    allow_descendant: bool = False,
) -> str:
    resolved = path.resolve()
    if source_root is not None:
        root = source_root.resolve()
        contained = root in resolved.parents if allow_descendant else resolved.parent == root
        if not contained:
            raise MediaTokenError("A linked media file resolves outside its configured source folder.")
    return _register(
        MediaAccess(
            path=resolved,
            kind=kind,
            source_root=source_root.resolve() if source_root is not None else None,
            caption_mode=caption_mode,
            caption_path=caption_path.resolve() if caption_path is not None else None,
            manifest_path=manifest_path.resolve() if manifest_path is not None else None,
            manifest_line=manifest_line,
            media_key=media_key,
            media_value=media_value,
            allow_descendant=allow_descendant,
        )
    )


def _directory_inventory(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root: Path = spec["path"]
    kind = spec["kind"]
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    extensions = VIDEO_EXTENSIONS if kind == "video" else IMAGE_EXTENSIONS
    paths = sorted(
        (
            path.resolve()
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in extensions and path.resolve().parent == root.resolve()
        ),
        key=_natural_key,
    )
    caption_extension = spec["caption_extension"]
    dataset = spec["dataset"]
    control_value = str(dataset.get("control_directory") or "").strip()
    control_root = resolve_configured_path(control_value) if control_value else None
    control_map = _directory_control_map(control_root, paths, kind)

    caption_stem_counts: dict[str, int] = {}
    for path in paths:
        caption_stem_counts[path.stem.casefold()] = caption_stem_counts.get(path.stem.casefold(), 0) + 1
    captioned_stems = {
        path.stem.casefold()
        for path in paths
        if caption_extension and path.with_suffix(caption_extension).is_file()
    }

    rows: list[dict[str, Any]] = []
    counts = {
        "media_count": len(paths),
        "primary_count": 0,
        "caption_count": 0,
        "missing_caption_count": 0,
        "empty_caption_count": 0,
        "unreadable_caption_count": 0,
        "trainer_usable_count": 0,
        "control_count": 0,
        "missing_control_count": 0,
        "shared_caption_count": sum(count - 1 for count in caption_stem_counts.values() if count > 1),
    }
    multiple_target = bool(dataset.get("multiple_target")) and kind == "image"
    for path in paths:
        caption_path = path.with_suffix(caption_extension) if caption_extension else None
        caption, caption_state, caption_revision = _read_caption(caption_path)
        role = "primary"
        if multiple_target and caption_state in {"missing", "not_configured"}:
            stem_parts = path.stem.rsplit("_", 1)
            if len(stem_parts) == 2 and stem_parts[1].isdigit() and stem_parts[0].casefold() in captioned_stems:
                role = "target"
        if role == "primary":
            counts["primary_count"] += 1
            if caption_state in {"present", "empty"}:
                counts["caption_count"] += 1
            if caption_state in {"missing", "not_configured"}:
                counts["missing_caption_count"] += 1
            elif caption_state == "empty":
                counts["empty_caption_count"] += 1
            elif caption_state in {"unreadable", "too_large"}:
                counts["unreadable_caption_count"] += 1

        if role == "target":
            training_state = "paired_target"
        elif caption_state in {"present", "empty"}:
            training_state = "eligible" if caption_state == "present" else "warning"
            counts["trainer_usable_count"] += 1
        elif kind == "image" and caption_state == "missing":
            training_state = "excluded"
        else:
            training_state = "error"

        controls = control_map.get(path, [])
        if control_root is not None:
            if controls:
                counts["control_count"] += 1
            elif role == "primary":
                counts["missing_control_count"] += 1

        rows.append(
            {
                "path": path,
                "name": path.name,
                "relative_path": path.name,
                "kind": kind,
                "preview_kind": kind,
                "bytes": path.stat().st_size,
                "caption": caption,
                "caption_state": caption_state,
                "caption_revision": caption_revision,
                "caption_mode": "sidecar",
                "training_state": training_state,
                "role": role,
                "shared_caption": caption_stem_counts[path.stem.casefold()] > 1,
                "controls": controls,
                "control_root": control_root,
                "missing_media": False,
            }
        )
    counts["effective_samples"] = counts["trainer_usable_count"] * spec["repeats"]
    counts["repeats"] = spec["repeats"]
    counts["control_configured"] = control_root is not None
    counts["control_root"] = str(control_root or "")
    return rows, counts


def _resolve_manifest_media(value: Any) -> Path:
    return resolve_configured_path(str(value or ""))


def _manifest_inventory(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest: Path = spec["path"]
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset JSONL file does not exist: {manifest}")
    kind = spec["kind"]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    manifest_revision = _file_revision(manifest)
    physical_lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    for line_index, raw_line in enumerate(physical_lines):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append({"line": line_index + 1, "message": str(exc)})
            continue
        key = "video_path" if kind == "video" else ("image_path" if "image_path" in record else "image_path_0")
        media_value = str(record.get(key) or "").strip()
        path = _resolve_manifest_media(media_value) if media_value else Path()
        caption_value = record.get("caption")
        caption = caption_value if isinstance(caption_value, str) else ""
        caption_state = (
            "missing"
            if "caption" not in record or not isinstance(caption_value, str)
            else ("empty" if not caption.strip() else "present")
        )
        target_values = [
            str(record[name])
            for name in sorted(record)
            if name.startswith("image_path_") and name != key and str(record[name] or "").strip()
        ]
        control_values = [
            str(record[name])
            for name in sorted(record)
            if (name == "control_path" or name.startswith("control_path_")) and str(record[name] or "").strip()
        ]
        missing_media = not media_value or not path.exists()
        preview_path = path
        preview_kind = kind
        if kind == "video" and path.is_dir():
            frames = sorted(
                (
                    item.resolve()
                    for item in path.iterdir()
                    if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=_natural_key,
            )
            if frames:
                preview_path = frames[0]
                preview_kind = "image"
            else:
                missing_media = True
        training_state = "error" if missing_media or caption_state == "missing" else (
            "warning" if caption_state == "empty" else "eligible"
        )
        rows.append(
            {
                "path": path,
                "preview_path": preview_path,
                "name": path.name if media_value else f"JSONL line {line_index + 1}",
                "relative_path": media_value,
                "kind": kind,
                "preview_kind": preview_kind,
                "bytes": path.stat().st_size if path.is_file() else 0,
                "caption": caption,
                "caption_state": caption_state,
                "caption_revision": manifest_revision,
                "caption_mode": "jsonl",
                "training_state": training_state,
                "role": "primary",
                "shared_caption": False,
                "controls": [_resolve_manifest_media(value) for value in control_values],
                "control_root": None,
                "targets": [_resolve_manifest_media(value) for value in target_values],
                "missing_media": missing_media,
                "manifest_line": line_index,
                "media_key": key,
                "media_value": media_value,
            }
        )
    trainer_usable = sum(row["training_state"] in {"eligible", "warning"} for row in rows)
    counts = {
        "media_count": len(rows),
        "primary_count": len(rows),
        "caption_count": sum(row["caption_state"] in {"present", "empty"} for row in rows),
        "missing_caption_count": sum(row["caption_state"] == "missing" for row in rows),
        "empty_caption_count": sum(row["caption_state"] == "empty" for row in rows),
        "unreadable_caption_count": 0,
        "trainer_usable_count": trainer_usable,
        "effective_samples": trainer_usable * spec["repeats"],
        "repeats": spec["repeats"],
        "control_count": sum(bool(row["controls"]) for row in rows),
        "missing_control_count": 0,
        "shared_caption_count": 0,
        "manifest_errors": errors,
        "manifest_revision": manifest_revision,
    }
    return rows, counts


def _matches_filter(row: dict[str, Any], mode: str) -> bool:
    if mode == "missing_caption":
        return row["caption_state"] == "missing"
    if mode == "needs_attention":
        return row["training_state"] in {"excluded", "error", "warning"}
    if mode == "eligible":
        return row["training_state"] in {"eligible", "warning"}
    if mode == "missing_media":
        return bool(row.get("missing_media"))
    if mode == "controls":
        return bool(row.get("controls"))
    return True


def list_dataset_media(
    text: str,
    source_path: str = "",
    index: int = 0,
    page: int = 1,
    page_size: int = 24,
    query: str = "",
    filter_mode: str = "all",
) -> dict[str, Any]:
    del source_path  # Paths intentionally resolve like the trainer, from cwd.
    spec = _source_spec(text, index)
    page = max(1, int(page or 1))
    page_size = max(1, min(MAX_PAGE_SIZE, int(page_size or 24)))
    rows, overview = (
        _directory_inventory(spec) if spec["mode"] == "directory" else _manifest_inventory(spec)
    )
    query_folded = str(query or "").strip().casefold()
    filtered = [
        row
        for row in rows
        if _matches_filter(row, filter_mode)
        and (
            not query_folded
            or query_folded in row["name"].casefold()
            or query_folded in row["relative_path"].casefold()
            or query_folded in row["caption"].casefold()
        )
    ]
    total = len(filtered)
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, pages)
    selected = filtered[(page - 1) * page_size : page * page_size]
    items = []
    root = spec["path"] if spec["mode"] == "directory" else None
    for row in selected:
        preview_path = row.get("preview_path") or row["path"]
        token = ""
        if preview_path.is_file():
            token = _register_preview(
                preview_path,
                kind=row["preview_kind"],
                source_root=root,
                caption_mode=row["caption_mode"],
                caption_path=(
                    row["path"].with_suffix(spec["caption_extension"])
                    if row["caption_mode"] == "sidecar" and spec["caption_extension"]
                    else None
                ),
                manifest_path=spec["path"] if row["caption_mode"] == "jsonl" else None,
                manifest_line=row.get("manifest_line"),
                media_key=row.get("media_key", ""),
                media_value=row.get("media_value", ""),
            )
        controls = []
        for control in row.get("controls", [])[:4]:
            control_path = control
            if control_path.is_dir():
                frames = sorted(
                    (
                        item.resolve()
                        for item in control_path.iterdir()
                        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
                    ),
                    key=_natural_key,
                )
                control_path = frames[0] if frames else control_path
            if not control_path.is_file():
                continue
            control_kind = "image" if control_path.suffix.lower() in IMAGE_EXTENSIONS else "video"
            try:
                controls.append(
                    {
                        "name": control_path.name,
                        "token": _register_preview(
                            control_path,
                            kind=control_kind,
                            source_root=row.get("control_root"),
                            allow_descendant=row.get("control_root") is not None,
                        ),
                    }
                )
            except (OSError, MediaTokenError):
                continue
        width = height = None
        media_error = ""
        if row["preview_kind"] == "image" and preview_path.is_file():
            width, height, media_error = _image_dimensions(preview_path)
        items.append(
            {
                "token": token,
                "name": row["name"],
                "relative_path": row["relative_path"],
                "kind": row["kind"],
                "preview_kind": row["preview_kind"],
                "bytes": row["bytes"],
                "width": width,
                "height": height,
                "media_error": media_error,
                "caption": row["caption"],
                "caption_state": row["caption_state"],
                "caption_revision": row["caption_revision"],
                "caption_mode": row["caption_mode"],
                "training_state": row["training_state"],
                "role": row["role"],
                "shared_caption": row["shared_caption"],
                "controls": controls,
                "target_count": len(row.get("targets", [])),
                "missing_media": row["missing_media"],
            }
        )
    return {
        "source": {
            "index": index,
            "kind": spec["kind"],
            "mode": spec["mode"],
            "path": str(spec["path"]),
            "caption_extension": spec["caption_extension"],
        },
        "overview": overview,
        "items": items,
        "page": page,
        "page_size": page_size,
        "pages": pages,
        "total": total,
        "filter": filter_mode,
        "query": query,
    }


def dataset_source_location(text: str, index: int) -> Path:
    spec = _source_spec(text, index)
    source = spec["path"].resolve()
    target = source.parent if source.is_file() else source
    if not target.is_dir():
        raise FileNotFoundError(f"Dataset source does not exist: {source}")
    return target


def audit_dataset_sources(text: str, source_path: str = "") -> dict[str, Any]:
    del source_path
    plain = plain_document(parse_document(text))
    datasets = plain.get("datasets", [])
    reports = []
    for index, _dataset in enumerate(datasets if isinstance(datasets, list) else []):
        report: dict[str, Any] = {
            "index": index,
            "media_count": 0,
            "caption_count": 0,
            "missing_caption_count": 0,
            "empty_caption_count": 0,
            "unreadable_caption_count": 0,
            "unreadable_count": 0,
            "trainer_usable_count": 0,
            "effective_samples": 0,
            "extensions": {},
            "resolutions": {},
            "aspects": {"landscape": 0, "portrait": 0, "square": 0},
            "examples_missing_caption": [],
            "ignored_nested_count": 0,
            "unsupported_count": 0,
        }
        try:
            spec = _source_spec(text, index)
            rows, overview = (
                _directory_inventory(spec) if spec["mode"] == "directory" else _manifest_inventory(spec)
            )
            report.update(
                {
                    "kind": spec["kind"],
                    "mode": spec["mode"],
                    "source": str(spec["path"]),
                    **overview,
                }
            )
            if spec["mode"] == "directory":
                root: Path = spec["path"]
                extensions = VIDEO_EXTENSIONS if spec["kind"] == "video" else IMAGE_EXTENSIONS
                report["ignored_nested_count"] = sum(
                    1
                    for path in root.rglob("*")
                    if path.is_file() and path.parent != root and path.suffix.lower() in extensions
                )
                caption_extension = spec["caption_extension"]
                report["unsupported_count"] = sum(
                    1
                    for path in root.iterdir()
                    if path.is_file()
                    and path.suffix.lower() not in extensions
                    and (not caption_extension or not path.name.lower().endswith(caption_extension.lower()))
                )
            for row in rows:
                path: Path = row["path"]
                if path.suffix:
                    suffix = path.suffix.lower()
                    report["extensions"][suffix] = report["extensions"].get(suffix, 0) + 1
                if row["caption_state"] == "missing" and len(report["examples_missing_caption"]) < 8:
                    report["examples_missing_caption"].append(str(path))
                preview_path = row.get("preview_path") or path
                if row["preview_kind"] != "image" or not preview_path.is_file():
                    continue
                width, height, error = _image_dimensions(preview_path)
                if error or width is None or height is None:
                    report["unreadable_count"] += 1
                    continue
                resolution = f"{width}×{height}"
                report["resolutions"][resolution] = report["resolutions"].get(resolution, 0) + 1
                aspect = "square" if width == height else ("landscape" if width > height else "portrait")
                report["aspects"][aspect] += 1
        except Exception as exc:
            report.setdefault("kind", "image")
            report["error"] = str(exc)
        reports.append(report)
    return {"datasets": reports}


def save_media_caption(token: str, caption: str, expected_revision: str) -> dict[str, Any]:
    access = resolve_media_token(token)
    encoded = str(caption).encode("utf-8")
    if len(encoded) > MAX_CAPTION_BYTES:
        raise ValueError(f"Caption is too large. The limit is {MAX_CAPTION_BYTES // 1024} KiB.")
    if access.caption_mode == "sidecar":
        path = access.caption_path
        if path is None or access.source_root is None:
            raise MediaTokenError("This item has no editable sidecar caption.")
        path = path.resolve()
        if path.parent != access.source_root.resolve():
            raise MediaTokenError("The caption resolves outside the dataset source.")
        current_revision = _file_revision(path)
        if current_revision != expected_revision:
            raise CaptionConflictError(
                "This caption changed on disk after you opened it. Refresh before overwriting it."
            )
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
        try:
            temporary.write_bytes(encoded)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {
            "caption": str(caption),
            "caption_state": "empty" if not str(caption).strip() else "present",
            "caption_revision": _file_revision(path),
            "caption_mode": "sidecar",
        }
    if access.caption_mode != "jsonl" or access.manifest_path is None or access.manifest_line is None:
        raise MediaTokenError("This item does not have an editable caption.")
    manifest = access.manifest_path.resolve()
    current_revision = _file_revision(manifest)
    if current_revision != expected_revision:
        raise CaptionConflictError(
            "This JSONL manifest changed after you opened it. Refresh before overwriting it."
        )
    lines = manifest.read_bytes().splitlines(keepends=True)
    if access.manifest_line < 0 or access.manifest_line >= len(lines):
        raise CaptionConflictError("The JSONL record moved. Refresh the dataset browser.")
    original = lines[access.manifest_line]
    newline = b"\r\n" if original.endswith(b"\r\n") else (b"\n" if original.endswith(b"\n") else b"")
    bom = b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
    record = json.loads(original.decode("utf-8-sig").rstrip("\r\n"))
    if str(record.get(access.media_key) or "") != access.media_value:
        raise CaptionConflictError("The JSONL media record changed. Refresh the dataset browser.")
    record["caption"] = str(caption)
    lines[access.manifest_line] = bom + json.dumps(record, ensure_ascii=False).encode("utf-8") + newline
    temporary = manifest.with_name(f".{manifest.name}.{secrets.token_hex(6)}.tmp")
    try:
        temporary.write_bytes(b"".join(lines))
        os.replace(temporary, manifest)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "caption": str(caption),
        "caption_state": "empty" if not str(caption).strip() else "present",
        "caption_revision": _file_revision(manifest),
        "caption_mode": "jsonl",
    }
