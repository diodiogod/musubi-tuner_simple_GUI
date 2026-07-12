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


def models_complete(model_dir: str | Path) -> bool:
    root = Path(model_dir).expanduser()
    return all((root / item.relative_path).is_file() and (root / item.relative_path).stat().st_size >= item.minimum_bytes for item in FACE_MODEL_FILES)


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
                    progress(destination.name, received, total)
        if partial.stat().st_size < item.minimum_bytes:
            raise RuntimeError(f"Downloaded face model is unexpectedly small: {partial}")
        partial.replace(destination)
    return root
