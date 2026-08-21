from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import toml

try:
    import tomlkit
except ImportError:  # pragma: no cover - the project declares tomlkit
    tomlkit = None


# Keep these aligned with musubi_tuner.dataset.media_utils without importing that
# module (it eagerly imports OpenCV and PyAV). Musubi scans direct children only.
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp", ".avif"}
if find_spec("jxlpy") is not None or find_spec("pillow_jxl") is not None:
    IMAGE_EXTENSIONS.add(".jxl")
VIDEO_EXTENSIONS = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}

ASCENDABLE_DEFAULTS = {
    "resolution": None,
    "num_repeats": 1,
    "batch_size": 1,
    "caption_extension": None,
    "enable_bucket": False,
    "bucket_no_upscale": False,
}
ASCENDABLE_KEYS = tuple(ASCENDABLE_DEFAULTS)
GENERAL_EDITABLE_KEYS = set(ASCENDABLE_KEYS)
COMMON_DATASET_KEYS = {
    *ASCENDABLE_KEYS,
    "cache_directory",
    "control_directory",
}
IMAGE_DATASET_KEYS = {
    "image_directory",
    "image_jsonl_file",
    "multiple_target",
    "no_resize_control",
    "control_resolution",
}
VIDEO_DATASET_KEYS = {
    "video_directory",
    "video_jsonl_file",
    "target_frames",
    "frame_extraction",
    "frame_stride",
    "frame_sample",
    "max_frames",
    "source_fps",
}

# Disabled sources are kept in the TOML as a clearly-owned, commented block.
# Ordinary comments remain untouched, and Musubi's TOML parser naturally ignores
# the block until the GUI explicitly re-enables it.
DISABLED_DATASET_START = "# musubi-gui: disabled dataset v1"
DISABLED_DATASET_END = "# musubi-gui: end disabled dataset"
DISABLED_DATASET_POSITION = "__musubi_gui_position"
DISABLED_DATASET_START_PATTERN = re.compile(
    r"^#\s*musubi-gui:\s*disabled dataset v1(?:\s+position=(\d+))?\s*$"
)


class DocumentConflictError(RuntimeError):
    """Raised when a disk document changed after the UI loaded it."""


@dataclass(frozen=True)
class DatasetIssue:
    level: str
    message: str
    dataset_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "message": self.message, "dataset_index": self.dataset_index}


def parse_document(text: str) -> Any:
    if tomlkit is not None:
        return tomlkit.parse(text)
    return toml.loads(text)


def dump_document(document: Any) -> str:
    if tomlkit is not None:
        return tomlkit.dumps(document)
    return toml.dumps(document)


def plain_document(document: Any) -> dict[str, Any]:
    return toml.loads(dump_document(document))


def _uncomment_disabled_line(line: str) -> str:
    """Remove the one comment marker added by ``_comment_disabled_block``."""

    if line.startswith("#"):
        line = line[1:]
        if line.startswith(" "):
            line = line[1:]
    return line


def _comment_disabled_block(text: str) -> str:
    lines = text.splitlines()
    commented = [f"# {line}" if line else "#" for line in lines]
    return "\n".join(commented) + ("\n" if commented else "")


def _split_disabled_datasets(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract GUI-owned commented dataset blocks before TOML parsing.

    The marker is deliberately explicit so a user's normal explanatory TOML
    comments are never interpreted as disabled sources. A malformed marked
    block is an error rather than silently losing a dataset.
    """

    active_lines: list[str] = []
    disabled: list[dict[str, Any]] = []
    lines = text.splitlines(keepends=True)
    cursor = 0
    while cursor < len(lines):
        marker = DISABLED_DATASET_START_PATTERN.match(lines[cursor].strip())
        if marker is None:
            active_lines.append(lines[cursor])
            cursor += 1
            continue
        position = int(marker.group(1)) if marker.group(1) is not None else None
        cursor += 1
        block_lines: list[str] = []
        found_end = False
        while cursor < len(lines):
            if lines[cursor].strip() == DISABLED_DATASET_END:
                found_end = True
                cursor += 1
                break
            block_lines.append(_uncomment_disabled_line(lines[cursor]))
            cursor += 1
        if not found_end:
            raise ValueError("A disabled dataset block is missing its end marker.")
        block_text = "".join(block_lines).strip()
        try:
            block_document = parse_document(block_text)
            block_plain = plain_document(block_document)
        except Exception as exc:
            raise ValueError("A disabled dataset block contains invalid TOML.") from exc
        entries = block_plain.get("datasets")
        if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
            raise ValueError("Each disabled dataset block must contain exactly one [[datasets]] entry.")
        dataset = _json_safe(entries[0])
        if position is not None:
            dataset[DISABLED_DATASET_POSITION] = position
        disabled.append(dataset)
    return "".join(active_lines), disabled


def _append_disabled_datasets(text: str, disabled: list[dict[str, Any]]) -> str:
    if not disabled:
        return text
    blocks = []
    for dataset in disabled:
        stored = _json_safe(dataset)
        position = stored.pop(DISABLED_DATASET_POSITION, None)
        serialized = toml.dumps({"datasets": [stored]}).rstrip()
        marker = f"{DISABLED_DATASET_START} position={int(position)}" if position is not None else DISABLED_DATASET_START
        blocks.append(f"{marker}\n{_comment_disabled_block(serialized)}{DISABLED_DATASET_END}\n")
    base = text.rstrip()
    return (base + "\n\n" if base else "") + "\n".join(blocks)


def _parse_document_with_disabled(text: str) -> tuple[Any, list[dict[str, Any]]]:
    active_text, disabled = _split_disabled_datasets(text)
    return parse_document(active_text), disabled


def _dump_document_with_disabled(document: Any, disabled: list[dict[str, Any]]) -> str:
    return _append_disabled_datasets(dump_document(document), disabled)


def _dataset_summary(dataset: dict[str, Any], index: int, general: dict[str, Any]) -> dict[str, Any]:
    kind = _dataset_kind(dataset)
    directory_key = f"{kind}_directory"
    jsonl_key = f"{kind}_jsonl_file"
    source_mode = "jsonl" if dataset.get(jsonl_key) else "directory"
    source_key = jsonl_key if source_mode == "jsonl" else directory_key
    source = dataset.get(source_key) or ""
    effective = {}
    origins = {}
    for key, default in ASCENDABLE_DEFAULTS.items():
        if key in dataset:
            effective[key] = dataset[key]
            origins[key] = "dataset"
        elif key in general:
            effective[key] = general[key]
            origins[key] = "general"
        else:
            effective[key] = default
            origins[key] = "default"
    known = COMMON_DATASET_KEYS | (VIDEO_DATASET_KEYS if kind == "video" else IMAGE_DATASET_KEYS)
    return {
        "index": index,
        "kind": kind,
        "source": str(source),
        "source_mode": source_mode,
        "source_key": source_key,
        "resolved_source": str(resolve_configured_path(source)) if source else "",
        "resolution": effective["resolution"],
        "repeats": effective["num_repeats"],
        "batch_size": effective["batch_size"],
        "cache_directory": dataset.get("cache_directory", ""),
        "caption_extension": effective["caption_extension"],
        "enable_bucket": effective["enable_bucket"],
        "bucket_no_upscale": effective["bucket_no_upscale"],
        "target_frames": dataset.get("target_frames", []),
        "raw_values": _json_safe(dataset),
        "effective_values": _json_safe(effective),
        "value_origins": origins,
        "inherited_from_general": [key for key, origin in origins.items() if origin == "general"],
        "advanced_keys": sorted(str(key) for key in set(dataset) - known),
        # Kept for backward-compatible clients. This is always raw.
        "values": _json_safe(dataset),
    }


def _drop_empty_cache_directories(document: Any) -> None:
    """Remove blank cache keys so Musubi can apply its source-directory fallback."""

    datasets = document.get("datasets") if isinstance(document, dict) else None
    if not isinstance(datasets, list):
        return
    for dataset in datasets:
        if isinstance(dataset, dict) and "cache_directory" in dataset:
            if not str(dataset.get("cache_directory") or "").strip():
                dataset.pop("cache_directory", None)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def document_revision(path: str) -> str:
    if not path:
        return ""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return ""
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def resolve_configured_path(value: Any) -> Path:
    """Resolve paths the same way a web-launched Musubi command does.

    Dataset paths are passed through unchanged to Musubi, whose process runs
    from the repository working directory. They are not relative to the TOML.
    """

    path = Path(str(value or "")).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _dataset_kind(dataset: dict[str, Any]) -> str:
    return "video" if "video_directory" in dataset or "video_jsonl_file" in dataset else "image"


def summarize_document(text: str, source_path: str = "") -> dict[str, Any]:
    document, disabled_datasets = _parse_document_with_disabled(text)
    plain = plain_document(document)
    datasets = plain.get("datasets", [])
    if not isinstance(datasets, list):
        datasets = []
    general = plain.get("general", {})
    if not isinstance(general, dict):
        general = {}
    summaries = [_dataset_summary(dataset, index, general) for index, dataset in enumerate(datasets)]
    disabled_summaries = []
    for disabled_index, dataset in enumerate(disabled_datasets):
        clean_dataset = {key: value for key, value in dataset.items() if key != DISABLED_DATASET_POSITION}
        summary = _dataset_summary(clean_dataset, len(summaries) + disabled_index, general)
        summary["disabled"] = True
        summary["disabled_index"] = disabled_index
        summary["position"] = int(dataset.get(DISABLED_DATASET_POSITION, len(summaries) + disabled_index))
        disabled_summaries.append(summary)
    return {
        "path": source_path,
        "text": _dump_document_with_disabled(document, disabled_datasets),
        "general": _json_safe(general),
        "datasets": summaries,
        "disabled_datasets": disabled_summaries,
        "issues": [issue.as_dict() for issue in validate_document(plain, source_path)],
        "disk_revision": document_revision(source_path),
        "preservation_available": tomlkit is not None,
    }


def validate_document(document: dict[str, Any], source_path: str = "") -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return [DatasetIssue("error", "The configuration needs at least one [[datasets]] entry.")]

    general = document.get("general", {})
    if not isinstance(general, dict):
        issues.append(DatasetIssue("error", "[general] must be a TOML table."))
        general = {}
    cache_owners: dict[str, int] = {}
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            issues.append(DatasetIssue("error", "Dataset entry is not a TOML table.", index))
            continue
        kind = _dataset_kind(dataset)
        source_keys = (
            "image_directory",
            "image_jsonl_file",
            "video_directory",
            "video_jsonl_file",
        )
        populated_sources = [key for key in source_keys if str(dataset.get(key) or "").strip()]
        if len(populated_sources) > 1:
            issues.append(
                DatasetIssue(
                    "error",
                    "Choose exactly one directory or JSONL source; multiple source fields are populated.",
                    index,
                )
            )
        directory = dataset.get(f"{kind}_directory")
        jsonl = dataset.get(f"{kind}_jsonl_file")
        if not directory and not jsonl:
            issues.append(DatasetIssue("error", f"Choose a {kind} directory or JSONL file.", index))
        if directory:
            path = resolve_configured_path(directory)
            if not path.is_dir():
                issues.append(DatasetIssue("warning", f"Directory does not currently exist: {path}", index))
        if jsonl:
            path = resolve_configured_path(jsonl)
            if not path.is_file():
                issues.append(DatasetIssue("warning", f"JSONL file does not currently exist: {path}", index))
        resolution = dataset.get("resolution", general.get("resolution"))
        if resolution is None:
            issues.append(DatasetIssue("error", "Resolution is required.", index))
        elif isinstance(resolution, bool) or not isinstance(resolution, (int, list, tuple)):
            issues.append(DatasetIssue("error", "Resolution must be a positive number or two dimensions.", index))
        elif isinstance(resolution, (list, tuple)) and (
            len(resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in resolution)
        ):
            issues.append(DatasetIssue("error", "Resolution must contain two positive dimensions.", index))
        elif isinstance(resolution, int) and resolution < 1:
            issues.append(DatasetIssue("error", "Resolution must be positive.", index))
        for key, label in (("num_repeats", "Repeats"), ("batch_size", "Batch size")):
            value = dataset.get(key, general.get(key))
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                issues.append(DatasetIssue("error", f"{label} must be a positive whole number.", index))
        cache = str(dataset.get("cache_directory") or "").strip()
        effective_cache = cache or str(directory or jsonl or "").strip()
        if effective_cache:
            resolved_cache = str(resolve_configured_path(effective_cache)).casefold()
            if resolved_cache in cache_owners:
                issues.append(
                    DatasetIssue(
                        "error",
                        f"Effective cache location is already used by Source {cache_owners[resolved_cache] + 1}.",
                        index,
                    )
                )
            else:
                cache_owners[resolved_cache] = index
        if jsonl and not cache:
            issues.append(DatasetIssue("warning", "Choose a dedicated cache directory for this JSONL source.", index))

        caption_extension = dataset.get("caption_extension", general.get("caption_extension"))
        if directory and not str(caption_extension or "").strip():
            issues.append(
                DatasetIssue(
                    "warning",
                    "Directory sources need a caption extension before captions can be loaded.",
                    index,
                )
            )

        if kind == "video":
            target_frames = dataset.get("target_frames")
            if not target_frames:
                issues.append(DatasetIssue("warning", "No target_frames are configured for this video dataset.", index))
            elif not isinstance(target_frames, list) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in target_frames
            ):
                issues.append(DatasetIssue("error", "Target frames must be positive whole numbers.", index))
            extraction = str(dataset.get("frame_extraction", "head"))
            if extraction not in {"head", "chunk", "slide", "uniform", "full"}:
                issues.append(DatasetIssue("error", f"Unknown frame extraction mode: {extraction}", index))
            if extraction == "chunk" and isinstance(target_frames, list) and 1 in target_frames:
                issues.append(
                    DatasetIssue("error", "Chunk extraction cannot be combined with target frame 1.", index)
                )
    return issues


def load_document(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return summarize_document(resolved.read_text(encoding="utf-8"), str(resolved))


def save_document(path: str, text: str, expected_revision: str | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not str(path or "").strip():
        raise ValueError("Choose a dataset TOML path before saving.")
    current_revision = document_revision(str(resolved))
    if expected_revision is not None and current_revision != expected_revision:
        raise DocumentConflictError(
            "The TOML changed on disk after it was loaded. Reload it or copy your draft before saving."
        )
    parsed, disabled_datasets = _parse_document_with_disabled(text)
    _drop_empty_cache_directories(parsed)
    normalized = _dump_document_with_disabled(parsed, disabled_datasets)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(normalized, encoding="utf-8")
    temporary.replace(resolved)
    return summarize_document(normalized, str(resolved))


def update_dataset(text: str, index: int, changes: dict[str, Any], source_path: str = "") -> dict[str, Any]:
    document, disabled_datasets = _parse_document_with_disabled(text)
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or index < 0 or index >= len(datasets):
        raise IndexError(f"Dataset index {index} does not exist.")
    dataset = datasets[index]
    allowed = {
        "image_directory",
        "image_jsonl_file",
        "video_directory",
        "video_jsonl_file",
        "control_directory",
        "cache_directory",
        "resolution",
        "num_repeats",
        "batch_size",
        "caption_extension",
        "enable_bucket",
        "bucket_no_upscale",
        "target_frames",
        "frame_extraction",
        "frame_stride",
        "frame_sample",
        "max_frames",
        "source_fps",
        "multiple_target",
        "no_resize_control",
        "control_resolution",
        "fp_latent_window_size",
        "fp_1f_clean_indices",
        "fp_1f_target_index",
        "fp_1f_no_post",
    }
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(f"Unsupported dataset fields: {', '.join(sorted(unknown))}")
    populated_source_keys = [
        key
        for key in ("image_directory", "image_jsonl_file", "video_directory", "video_jsonl_file")
        if key in changes and str(changes[key] or "").strip()
    ]
    if len(populated_source_keys) > 1:
        raise ValueError("Choose one directory or JSONL source.")
    if populated_source_keys:
        selected_source = populated_source_keys[0]
        if selected_source.startswith("image_"):
            dataset.pop("video_directory", None)
            dataset.pop("video_jsonl_file", None)
            dataset.pop("image_jsonl_file" if selected_source == "image_directory" else "image_directory", None)
        else:
            dataset.pop("image_directory", None)
            dataset.pop("image_jsonl_file", None)
            dataset.pop("video_jsonl_file" if selected_source == "video_directory" else "video_directory", None)
    for key, value in changes.items():
        if value in ("", None):
            dataset.pop(key, None)
        else:
            dataset[key] = value
    return summarize_document(_dump_document_with_disabled(document, disabled_datasets), source_path)


def update_general(text: str, changes: dict[str, Any], source_path: str = "") -> dict[str, Any]:
    document, disabled_datasets = _parse_document_with_disabled(text)
    unknown = set(changes) - GENERAL_EDITABLE_KEYS
    if unknown:
        raise ValueError(f"Unsupported general fields: {', '.join(sorted(unknown))}")
    general = document.get("general")
    if general is None:
        general = tomlkit.table() if tomlkit is not None else {}
        document["general"] = general
    if not isinstance(general, dict):
        raise ValueError("[general] must be a TOML table.")
    for key, value in changes.items():
        if value in ("", None):
            general.pop(key, None)
        else:
            general[key] = value
    return summarize_document(_dump_document_with_disabled(document, disabled_datasets), source_path)


def add_dataset(
    text: str,
    kind: str,
    source_path: str = "",
    architecture: str = "",
    folder_path: str = "",
) -> dict[str, Any]:
    if kind not in {"image", "video"}:
        raise ValueError("Dataset kind must be image or video.")
    document, disabled_datasets = _parse_document_with_disabled(text) if text.strip() else (parse_document(""), [])
    if "datasets" not in document:
        if tomlkit is not None:
            document["datasets"] = tomlkit.aot()
        else:
            document["datasets"] = []
    _drop_empty_cache_directories(document)
    dataset = tomlkit.table() if tomlkit is not None else {}
    # Keep the source path separate from the TOML document path.  The former
    # comes from the native Explorer drop window; the latter is used for disk
    # revision tracking and must remain the path of the dataset TOML itself.
    dataset[f"{kind}_directory"] = str(folder_path or "")
    # Omit cache_directory when no dedicated cache is requested.  Musubi uses
    # the image/video source directory as its fallback; writing an explicit
    # empty string would bypass that fallback and resolve caches incorrectly.
    # Let fields already declared in [general] flow into the new source.  If
    # the document has no such declaration, preserve the editor's historical
    # defaults so a newly added source is still immediately usable.
    general = document.get("general", {})
    if not isinstance(general, dict):
        general = {}
    for key, fallback in (
        ("resolution", [1024, 1024]),
        ("num_repeats", 1),
        ("enable_bucket", True),
    ):
        if key not in general:
            dataset[key] = fallback
    if kind == "video":
        dataset["target_frames"] = [124] if architecture == "minimax_h3" else [1, 25]
    document["datasets"].append(dataset)
    return summarize_document(_dump_document_with_disabled(document, disabled_datasets), source_path)


def add_datasets(
    text: str,
    kind: str,
    folder_paths: list[str],
    source_path: str = "",
    architecture: str = "",
) -> dict[str, Any]:
    """Add several dropped folders while returning one final document summary."""

    paths = list(dict.fromkeys(str(path) for path in folder_paths if str(path or "").strip()))
    if not paths:
        return add_dataset(text, kind, source_path, architecture)
    current_text = text
    summary: dict[str, Any] | None = None
    for folder_path in paths:
        summary = add_dataset(current_text, kind, source_path, architecture, folder_path)
        current_text = summary["text"]
    assert summary is not None
    return summary


def split_dataset_subfolders(text: str, index: int, source_path: str = "") -> dict[str, Any]:
    """Expand a directory source into valid immediate child-folder sources.

    Musubi scans media sources directly, not recursively. Child folders with
    no supported media are ignored. If the selected parent has no direct media,
    it is replaced so nested files cannot remain silently ignored.
    """
    document, disabled_datasets = _parse_document_with_disabled(text)
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or index < 0 or index >= len(datasets):
        raise IndexError(f"Dataset index {index} does not exist.")
    dataset = datasets[index]
    if not isinstance(dataset, dict):
        raise ValueError("The selected dataset is not a TOML table.")

    kind = _dataset_kind(dataset)
    directory_key = f"{kind}_directory"
    jsonl_key = f"{kind}_jsonl_file"
    source = str(dataset.get(directory_key) or "").strip()
    if not source or str(dataset.get(jsonl_key) or "").strip():
        raise ValueError("Subfolder expansion requires a media-folder source, not a JSONL manifest.")
    root = resolve_configured_path(source)
    if not root.is_dir():
        raise ValueError(f"The selected source folder does not exist: {root}")

    extensions = IMAGE_EXTENSIONS if kind == "image" else VIDEO_EXTENSIONS
    direct_media = [item for item in root.iterdir() if item.is_file() and item.suffix.lower() in extensions]
    candidates: list[tuple[Path, int]] = []
    for child in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.casefold()):
        if child.name.startswith("."):
            continue
        media_count = sum(
            1 for item in child.iterdir() if item.is_file() and item.suffix.lower() in extensions
        )
        if media_count:
            candidates.append((child, media_count))
    if not candidates:
        raise ValueError(f"No immediate subfolders containing supported {kind} media were found in {root}.")

    def child_path(base: str, name: str) -> str:
        separator = "\\" if "\\" in base and "/" not in base else "/"
        return f"{base.rstrip('/\\')}{separator}{name}"

    existing_sources = {
        resolve_configured_path(str(item.get(f"{_dataset_kind(item)}_directory") or ""))
        for item in datasets
        if isinstance(item, dict) and str(item.get(f"{_dataset_kind(item)}_directory") or "").strip()
    }
    new_children: list[tuple[dict[str, Any], Path, int]] = []
    cache_directory = str(dataset.get("cache_directory") or "").strip()
    for child, media_count in candidates:
        child_source = child_path(source, child.name)
        if resolve_configured_path(child_source) in existing_sources:
            continue
        child_dataset = copy.deepcopy(dataset)
        child_dataset[directory_key] = child_source
        child_dataset.pop(jsonl_key, None)
        if cache_directory:
            child_dataset["cache_directory"] = child_path(cache_directory, child.name)
        else:
            child_dataset.pop("cache_directory", None)
        new_children.append((child_dataset, child, media_count))

    if not new_children:
        raise ValueError("All valid subfolders are already present in this TOML.")

    parent_has_direct_media = bool(direct_media)
    if parent_has_direct_media:
        insertion = index + 1
        for offset, (child_dataset, _child, _count) in enumerate(new_children):
            datasets.insert(insertion + offset, child_dataset)
        selected_index = insertion
    else:
        datasets[index] = new_children[0][0]
        for offset, (child_dataset, _child, _count) in enumerate(new_children[1:], start=1):
            datasets.insert(index + offset, child_dataset)
        selected_index = index

    summary = summarize_document(_dump_document_with_disabled(document, disabled_datasets), source_path)
    summary["subfolder_scan"] = {
        "parent": source,
        "kind": kind,
        "added": [
            {"path": child_path(source, child.name), "media_count": media_count}
            for _dataset, child, media_count in new_children
        ],
        "skipped_existing": len(candidates) - len(new_children),
        "removed_parent": not parent_has_direct_media,
        "selected_index": selected_index,
    }
    return summary


def _dataset_node_from_plain(dataset: dict[str, Any]) -> Any:
    """Turn a plain disabled record back into a TOML table for an AoT."""

    if tomlkit is None:
        return copy.deepcopy(dataset)
    wrapper = tomlkit.document()
    wrapper["datasets"] = tomlkit.aot()
    table = tomlkit.table()
    for key, value in dataset.items():
        table[key] = copy.deepcopy(value)
    wrapper["datasets"].append(table)
    return tomlkit.parse(tomlkit.dumps(wrapper))["datasets"][0]


def _plain_dataset(dataset: Any) -> dict[str, Any]:
    """Convert a TOMLKit table (including its typed scalar wrappers) to plain values."""

    if tomlkit is None:
        return copy.deepcopy(dataset)
    wrapper = tomlkit.document()
    wrapper["datasets"] = tomlkit.aot()
    wrapper["datasets"].append(copy.deepcopy(dataset))
    return toml.loads(tomlkit.dumps(wrapper))["datasets"][0]


def toggle_dataset_disabled(
    text: str,
    index: int,
    disabled: bool,
    source_path: str = "",
    position: int | None = None,
) -> dict[str, Any]:
    """Disable or re-enable one source without deleting its TOML settings.

    Disabled entries are serialized as GUI-owned comments at the end of the
    document. Re-enabling appends the source after the active entries; the
    source's complete settings and unknown fields are retained.
    """

    document, disabled_datasets = _parse_document_with_disabled(text)
    datasets = document.get("datasets")
    if datasets is None and not disabled:
        datasets = tomlkit.aot() if tomlkit is not None else []
        document["datasets"] = datasets
    if not isinstance(datasets, list):
        raise ValueError("The configuration has no datasets list.")
    if disabled:
        if index < 0 or index >= len(datasets):
            raise IndexError(f"Dataset index {index} does not exist.")
        removed = _plain_dataset(datasets.pop(index))
        removed[DISABLED_DATASET_POSITION] = max(0, int(position if position is not None else index))
        disabled_datasets.append(removed)
        action = "disabled"
        active_index = None
    else:
        if index < 0 or index >= len(disabled_datasets):
            raise IndexError(f"Disabled dataset index {index} does not exist.")
        restored = disabled_datasets.pop(index)
        restored_position = max(0, int(restored.pop(DISABLED_DATASET_POSITION, position or 0)))
        disabled_before = sum(
            1
            for item in disabled_datasets
            if int(item.get(DISABLED_DATASET_POSITION, len(datasets) + len(disabled_datasets))) < restored_position
        )
        active_position = max(0, min(restored_position - disabled_before, len(datasets)))
        datasets.insert(active_position, _dataset_node_from_plain(restored))
        action = "enabled"
        active_index = active_position
    summary = summarize_document(_dump_document_with_disabled(document, disabled_datasets), source_path)
    summary["disabled_action"] = {"action": action, "index": index, "active_index": active_index}
    return summary


def remove_dataset(text: str, index: int, source_path: str = "") -> dict[str, Any]:
    document, disabled_datasets = _parse_document_with_disabled(text)
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or index < 0 or index >= len(datasets):
        raise IndexError(f"Dataset index {index} does not exist.")
    del datasets[index]
    return summarize_document(_dump_document_with_disabled(document, disabled_datasets), source_path)


def duplicate_dataset(text: str, index: int, source_path: str = "") -> dict[str, Any]:
    import copy

    document, disabled_datasets = _parse_document_with_disabled(text)
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or index < 0 or index >= len(datasets):
        raise IndexError(f"Dataset index {index} does not exist.")
    duplicated = copy.deepcopy(datasets[index])
    if tomlkit is not None:
        # Parsed tables retain formatting separators that cannot safely be
        # inserted into the same AoT twice. Reparse only the copied table.
        wrapper = tomlkit.document()
        wrapper["datasets"] = tomlkit.aot()
        wrapper["datasets"].append(duplicated)
        duplicated = tomlkit.parse(tomlkit.dumps(wrapper))["datasets"][0]
    # Musubi cache directories must be unique. Match the classic editor's
    # safe copy behavior and force the user to choose a destination.
    duplicated.pop("cache_directory", None)
    datasets.insert(index + 1, duplicated)
    return summarize_document(_dump_document_with_disabled(document, disabled_datasets), source_path)


def move_dataset(text: str, index: int, destination: int, source_path: str = "") -> dict[str, Any]:
    document, disabled_datasets = _parse_document_with_disabled(text)
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or index < 0 or index >= len(datasets):
        raise IndexError(f"Dataset index {index} does not exist.")
    if destination < 0 or destination >= len(datasets):
        raise IndexError(f"Dataset destination {destination} does not exist.")
    if destination != index:
        dataset = datasets.pop(index)
        datasets.insert(destination, dataset)
    return summarize_document(_dump_document_with_disabled(document, disabled_datasets), source_path)


def inspect_dataset_sources(text: str, source_path: str = "") -> dict[str, Any]:
    from modern_gui.dataset_media import audit_dataset_sources

    return audit_dataset_sources(text, source_path)
