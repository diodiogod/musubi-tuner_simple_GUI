from __future__ import annotations

import json
import random
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from musubi_tuner.dataset.audio_utils import AUDIO_SIDECAR_EXTENSIONS, probe_audio, resolve_audio_source
from musubi_tuner.dataset.media_utils import VIDEO_EXTENSIONS
from musubi_tuner.minimax_h3_native.media import load_h3_jsonl_records, probe_h3_media

from modern_gui.dataset_documents import IMAGE_EXTENSIONS


def _files(directory: str, extensions: set[str] | frozenset[str]) -> list[Path]:
    root = Path(str(directory or "")).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Folder does not exist: {root}")
    return sorted(
        (path.resolve() for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.name.casefold(),
    )


def inspect_pairing_sources(image_directory: str, audio_directory: str) -> dict[str, Any]:
    images = _files(image_directory, IMAGE_EXTENSIONS)
    audio_files = _files(audio_directory, AUDIO_SIDECAR_EXTENSIONS)
    audio = []
    for path in audio_files:
        try:
            info = probe_h3_media(path)
            valid = probe_audio(path)
            audio.append({"path": str(path), "name": path.name, "duration": info.duration_seconds, "valid": valid})
        except Exception as exc:
            audio.append({"path": str(path), "name": path.name, "duration": None, "valid": False, "error": str(exc)})
    return {
        "images": [{"path": str(path), "name": path.name} for path in images],
        "audio": audio,
        "image_count": len(images),
        "audio_count": len(audio),
        "usable_audio_count": sum(item["valid"] for item in audio),
    }


def pairing_plan(
    image_directory: str,
    audio_directory: str,
    *,
    strategy: str = "round_robin",
    seed: int = 42,
) -> list[dict[str, Any]]:
    inspection = inspect_pairing_sources(image_directory, audio_directory)
    images = [Path(item["path"]) for item in inspection["images"]]
    audio = [item for item in inspection["audio"] if item["valid"]]
    if not images:
        raise ValueError("The image folder contains no supported still images.")
    if not audio:
        raise ValueError("The audio folder contains no decodable audio files.")
    if strategy not in {"round_robin", "random", "matching_stem"}:
        raise ValueError(f"Unsupported image pairing strategy: {strategy}")
    rng = random.Random(int(seed))
    image_by_stem = {path.stem.casefold(): path for path in images}
    plan = []
    for index, item in enumerate(audio):
        audio_path = Path(item["path"])
        if strategy == "matching_stem":
            image = image_by_stem.get(audio_path.stem.casefold())
            if image is None:
                raise ValueError(f"No same-name image exists for audio file {audio_path.name}")
        elif strategy == "random":
            image = rng.choice(images)
        else:
            image = images[index % len(images)]
        plan.append(
            {
                "audio_path": str(audio_path),
                "audio_name": audio_path.name,
                "image_path": str(image),
                "image_name": image.name,
                "duration": item["duration"],
                "output_name": f"{audio_path.stem}.mp4",
            }
        )
    return plan


def build_image_audio_videos(
    image_directory: str,
    audio_directory: str,
    output_directory: str,
    *,
    strategy: str = "round_robin",
    seed: int = 42,
    width: int = 768,
    height: int = 768,
    target_frames: int = 124,
    allow_experimental_duration: bool = False,
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to build image + audio training clips but was not found on PATH.")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("MiniMax H3 output width and height must be positive multiples of 32.")
    target_frames = int(target_frames)
    if target_frames < 5 or (target_frames - 5) % 17:
        raise ValueError("MiniMax H3 target frames must use 5 + 17×N (for example 124, 141, or 158).")
    if not allow_experimental_duration and not 124 <= target_frames <= 345:
        raise ValueError("Released MiniMax H3 training uses 124–345 frames. Enable experimental durations to go outside it.")
    target_seconds = target_frames / 24.0
    output = Path(str(output_directory or "")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = pairing_plan(image_directory, audio_directory, strategy=strategy, seed=seed)
    built = []
    for item in plan:
        duration = item["duration"]
        if duration is None:
            raise ValueError(f"Could not determine audio duration: {item['audio_name']}")
        if not allow_experimental_duration and not 5.0 <= duration <= 15.0:
            raise ValueError(
                f"{item['audio_name']} is {duration:.2f}s; released H3 training clips must be 5–15 seconds. "
                "Enable experimental durations only for a deliberate test."
            )
        destination = output / item["output_name"]
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite an existing generated clip: {destination}")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-loop", "1", "-framerate", "24",
            "-i", item["image_path"], "-i", item["audio_path"], "-map", "0:v:0", "-map", "1:a:0",
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-r", "24",
            "-frames:v", str(target_frames), "-af", "apad", "-t", f"{target_seconds:.6f}",
            "-c:a", "aac", "-ar", "32000", "-ac", "2", str(destination),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        audio_caption = Path(item["audio_path"]).with_suffix(".txt")
        image_caption = Path(item["image_path"]).with_suffix(".txt")
        caption_source = audio_caption if audio_caption.is_file() else image_caption if image_caption.is_file() else None
        caption_path = destination.with_suffix(".txt")
        caption_path.write_text(
            caption_source.read_text(encoding="utf-8-sig") if caption_source else "",
            encoding="utf-8",
        )
        built.append({
            **item,
            "video_path": str(destination),
            "caption_path": str(caption_path),
            "target_frames": target_frames,
            "target_duration": target_seconds,
        })
    return {
        "output_directory": str(output), "built": built, "count": len(built),
        "target_frames": target_frames, "target_duration": target_seconds,
    }


def load_ref2va_document(path: str) -> dict[str, Any]:
    manifest = Path(str(path or "")).expanduser().resolve()
    records = []
    if manifest.is_file():
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"Ref2VA line {line_number} must be a JSON object.")
            records.append(data)
    return {"path": str(manifest), "records": records}


def save_ref2va_document(path: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = Path(str(path or "")).expanduser().resolve()
    if manifest.suffix.lower() != ".jsonl":
        raise ValueError("Ref2VA manifest must use the .jsonl extension.")
    if not isinstance(records, list) or not records:
        raise ValueError("Add at least one Ref2VA target record before saving.")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    try:
        validated = load_h3_jsonl_records(temporary, "ref2va")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(manifest)
    return {"path": str(manifest), "records": records, "count": len(validated)}


def audit_h3_training_dataset(
    dataset_config: str,
    *,
    task: str,
    training_target: str,
    allow_experimental_duration: bool = False,
) -> dict[str, list[str]]:
    path = Path(str(dataset_config or "")).expanduser().resolve()
    if not path.is_file():
        return {"errors": [], "warnings": []}
    document = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    general = document.get("general") or {}
    datasets = document.get("datasets") or []
    errors: list[str] = []
    warnings: list[str] = []
    for index, dataset in enumerate(datasets, start=1):
        if not isinstance(dataset, dict):
            continue
        frames = dataset.get("target_frames", general.get("target_frames", []))
        if not isinstance(frames, list) or not frames:
            errors.append(f"Source {index} needs target_frames; use [124] for the released five-second H3 default.")
        else:
            for frame_count in frames:
                if not isinstance(frame_count, int) or frame_count < 5 or (frame_count - 5) % 17:
                    errors.append(f"Source {index} target frame count {frame_count!r} is invalid; H3 requires 17*n+5.")
                elif not allow_experimental_duration and not 124 <= frame_count <= 345:
                    errors.append(f"Source {index} frame count {frame_count} is outside released H3 duration 124–345.")
        directory = str(dataset.get("video_directory") or "").strip()
        manifest = str(dataset.get("video_jsonl_file") or "").strip()
        if task == "ref2va" and not manifest:
            errors.append(f"Source {index} must use video_jsonl_file for Ref2VA.")
            continue
        if manifest:
            try:
                records = load_h3_jsonl_records(manifest, task)
                if training_target.startswith("Audio only"):
                    manifest_path = Path(manifest).expanduser().resolve()
                    raw_records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
                    for record, raw in zip(records, raw_records):
                        explicit = raw.get("audio_path")
                        if explicit:
                            explicit_path = Path(explicit)
                            if not explicit_path.is_absolute():
                                explicit_path = manifest_path.parent / explicit_path
                        else:
                            explicit_path = None
                        if resolve_audio_source(record.video_path, explicit_path) is None:
                            errors.append(f"Audio-only target has no real audio: {record.video_path.name}")
            except Exception as exc:
                errors.append(str(exc))
            continue
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            continue
        videos = [item for item in root.iterdir() if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS]
        with_audio = 0
        for video in videos:
            try:
                if resolve_audio_source(video) is not None:
                    with_audio += 1
                elif training_target.startswith("Audio only"):
                    errors.append(f"Audio-only target has no real audio: {video.name}")
            except Exception as exc:
                errors.append(f"{video.name}: {exc}")
        if not training_target.startswith("Video only") and videos and with_audio < len(videos):
            warnings.append(f"Source {index}: {len(videos)-with_audio} of {len(videos)} clips have no audio supervision.")
    return {"errors": errors, "warnings": warnings}
