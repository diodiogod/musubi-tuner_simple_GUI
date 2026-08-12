from pathlib import Path

import pytest

from modern_gui import h3_datasets


def test_pairing_plan_rotates_images_and_ignores_bad_audio(monkeypatch, tmp_path: Path):
    images = tmp_path / "images"
    audio = tmp_path / "audio"
    images.mkdir()
    audio.mkdir()
    for name in ("one.png", "two.jpg"):
        (images / name).write_bytes(b"image")
    for name in ("a.wav", "b.mp3", "bad.wav"):
        (audio / name).write_bytes(b"audio")

    monkeypatch.setattr(
        h3_datasets,
        "probe_h3_media",
        lambda path: type("Info", (), {"duration_seconds": 6.0})(),
    )
    monkeypatch.setattr(h3_datasets, "probe_audio", lambda path: path.name != "bad.wav")

    plan = h3_datasets.pairing_plan(str(images), str(audio), strategy="round_robin")

    assert [item["image_name"] for item in plan] == ["one.png", "two.jpg"]
    assert [item["audio_name"] for item in plan] == ["a.wav", "b.mp3"]


def test_pairing_plan_matching_stem_requires_an_image(monkeypatch, tmp_path: Path):
    images = tmp_path / "images"
    audio = tmp_path / "audio"
    images.mkdir()
    audio.mkdir()
    (images / "someone.png").write_bytes(b"image")
    (audio / "voice.wav").write_bytes(b"audio")
    monkeypatch.setattr(h3_datasets, "probe_h3_media", lambda path: type("Info", (), {"duration_seconds": 6.0})())
    monkeypatch.setattr(h3_datasets, "probe_audio", lambda path: True)

    with pytest.raises(ValueError, match="same-name image"):
        h3_datasets.pairing_plan(str(images), str(audio), strategy="matching_stem")


def test_build_pairing_refuses_to_overwrite(monkeypatch, tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "voice.mp4").write_bytes(b"existing")
    monkeypatch.setattr(h3_datasets.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        h3_datasets,
        "pairing_plan",
        lambda *args, **kwargs: [{
            "audio_path": str(tmp_path / "voice.wav"), "audio_name": "voice.wav",
            "image_path": str(tmp_path / "face.png"), "image_name": "face.png",
            "duration": 6.0, "output_name": "voice.mp4",
        }],
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        h3_datasets.build_image_audio_videos("images", "audio", str(output))


def test_ref2va_save_validates_before_replacing(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "refs.jsonl"
    record = {"video_path": "target.mp4", "caption": "caption", "references": [{"type": "image", "path": "ref.png"}]}
    monkeypatch.setattr(h3_datasets, "load_h3_jsonl_records", lambda path, task: [object()])

    result = h3_datasets.save_ref2va_document(str(manifest), [record])

    assert result["count"] == 1
    assert h3_datasets.load_ref2va_document(str(manifest))["records"] == [record]


def test_h3_dataset_audit_checks_geometry(monkeypatch, tmp_path: Path):
    videos = tmp_path / "videos"
    videos.mkdir()
    config = tmp_path / "dataset.toml"
    config.write_text(
        f'[[datasets]]\nvideo_directory = "{videos.as_posix()}"\ntarget_frames = [25]\n',
        encoding="utf-8",
    )

    result = h3_datasets.audit_h3_training_dataset(
        str(config), task="t2va", training_target="Video only"
    )

    assert any("17*n+5" in message for message in result["errors"])
