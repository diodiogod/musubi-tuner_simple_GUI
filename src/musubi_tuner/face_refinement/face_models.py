"""Locate and download user-opted-in AntelopeV2 face-model files.

The model artifacts are not part of this repository and have separate terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FaceModelFile:
    relative_path: Path
    url: str
    minimum_bytes: int


@dataclass(frozen=True)
class FaceModelPaths:
    recognition: Path
    detection: Path


FACE_MODEL_FILES = (
    FaceModelFile(
        Path("recognition/model.onnx"),
        "https://huggingface.co/immich-app/antelopev2/resolve/main/recognition/model.onnx?download=true",
        200_000_000,
    ),
    FaceModelFile(
        Path("detection/model.onnx"),
        "https://huggingface.co/immich-app/antelopev2/resolve/main/detection/model.onnx?download=true",
        10_000_000,
    ),
)


def default_model_dir() -> Path:
    return Path.home() / ".cache" / "musubi-tuner" / "antelopev2"


def resolve_model_paths(model_dir: str | Path) -> FaceModelPaths:
    """Resolve either the GUI download layout or a standard InsightFace AntelopeV2 folder."""

    root = Path(model_dir).expanduser()
    roots = (root, root / "antelopev2")
    layouts = (
        (Path("recognition/model.onnx"), Path("detection/model.onnx")),
        (Path("glintr100.onnx"), Path("scrfd_10g_bnkps.onnx")),
    )
    recognition_minimum = FACE_MODEL_FILES[0].minimum_bytes
    detection_minimum = FACE_MODEL_FILES[1].minimum_bytes
    for candidate_root in roots:
        for recognition_name, detection_name in layouts:
            recognition = candidate_root / recognition_name
            detection = candidate_root / detection_name
            if (
                recognition.is_file()
                and recognition.stat().st_size >= recognition_minimum
                and detection.is_file()
                and detection.stat().st_size >= detection_minimum
            ):
                return FaceModelPaths(recognition=recognition, detection=detection)
    raise FileNotFoundError(
        "AntelopeV2 recognition and detection models were not found. Select either the GUI-downloaded "
        "folder containing recognition/model.onnx and detection/model.onnx, or a standard InsightFace "
        "folder containing glintr100.onnx and scrfd_10g_bnkps.onnx."
    )


def models_complete(model_dir: str | Path) -> bool:
    try:
        resolve_model_paths(model_dir)
        return True
    except (FileNotFoundError, OSError):
        return False


def ensure_models(model_dir: str | Path, progress=None) -> Path:
    root = Path(model_dir).expanduser().resolve()
    for item in FACE_MODEL_FILES:
        destination = root / item.relative_path
        if destination.is_file() and destination.stat().st_size >= item.minimum_bytes:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        request = Request(item.url, headers={"User-Agent": "musubi-tuner-face-refinement/1"})
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if progress:
                    progress(item.relative_path.as_posix(), received, total)
        if partial.stat().st_size < item.minimum_bytes:
            raise RuntimeError(f"Downloaded face model is unexpectedly small: {partial}")
        partial.replace(destination)
    return root
