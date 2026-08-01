from __future__ import annotations

import hashlib
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
    document = parse_document(text)
    plain = plain_document(document)
    datasets = plain.get("datasets", [])
    if not isinstance(datasets, list):
        datasets = []
    general = plain.get("general", {})
    if not isinstance(general, dict):
        general = {}
    summaries = []
    for index, dataset in enumerate(datasets):
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
        summaries.append(
            {
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
        )
    return {
        "path": source_path,
        "text": dump_document(document),
        "general": _json_safe(general),
        "datasets": summaries,
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
    parsed = parse_document(text)
    normalized = dump_document(parsed)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(normalized, encoding="utf-8")
    temporary.replace(resolved)
    return summarize_document(normalized, str(resolved))


def update_dataset(text: str, index: int, changes: dict[str, Any], source_path: str = "") -> dict[str, Any]:
    document = parse_document(text)
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
    return summarize_document(dump_document(document), source_path)


def update_general(text: str, changes: dict[str, Any], source_path: str = "") -> dict[str, Any]:
    document = parse_document(text)
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
    return summarize_document(dump_document(document), source_path)


def add_dataset(text: str, kind: str, source_path: str = "") -> dict[str, Any]:
    if kind not in {"image", "video"}:
        raise ValueError("Dataset kind must be image or video.")
    document = parse_document(text) if text.strip() else parse_document("")
    if "datasets" not in document:
        if tomlkit is not None:
            document["datasets"] = tomlkit.aot()
        else:
            document["datasets"] = []
    dataset = tomlkit.table() if tomlkit is not None else {}
    dataset[f"{kind}_directory"] = ""
    dataset["cache_directory"] = ""
    dataset["resolution"] = [1024, 1024]
    dataset["num_repeats"] = 1
    dataset["enable_bucket"] = True
    if kind == "video":
        dataset["target_frames"] = [1, 25]
    document["datasets"].append(dataset)
    return summarize_document(dump_document(document), source_path)


def remove_dataset(text: str, index: int, source_path: str = "") -> dict[str, Any]:
    document = parse_document(text)
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or index < 0 or index >= len(datasets):
        raise IndexError(f"Dataset index {index} does not exist.")
    del datasets[index]
    return summarize_document(dump_document(document), source_path)


def duplicate_dataset(text: str, index: int, source_path: str = "") -> dict[str, Any]:
    import copy

    document = parse_document(text)
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
    return summarize_document(dump_document(document), source_path)


def move_dataset(text: str, index: int, destination: int, source_path: str = "") -> dict[str, Any]:
    document = parse_document(text)
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or index < 0 or index >= len(datasets):
        raise IndexError(f"Dataset index {index} does not exist.")
    if destination < 0 or destination >= len(datasets):
        raise IndexError(f"Dataset destination {destination} does not exist.")
    if destination != index:
        dataset = datasets.pop(index)
        datasets.insert(destination, dataset)
    return summarize_document(dump_document(document), source_path)


def inspect_dataset_sources(text: str, source_path: str = "") -> dict[str, Any]:
    from modern_gui.dataset_media import audit_dataset_sources

    return audit_dataset_sources(text, source_path)
