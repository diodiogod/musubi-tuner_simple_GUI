from pathlib import Path

import pytest

from musubi_tuner.face_refinement import face_models


@pytest.fixture(autouse=True)
def _small_test_models(monkeypatch):
    monkeypatch.setattr(
        face_models,
        "FACE_MODEL_FILES",
        (
            face_models.FaceModelFile(Path("recognition/model.onnx"), "recognition", 1),
            face_models.FaceModelFile(Path("detection/model.onnx"), "detection", 1),
        ),
    )


def test_resolves_gui_download_layout(tmp_path: Path):
    recognition = tmp_path / "recognition" / "model.onnx"
    detection = tmp_path / "detection" / "model.onnx"
    recognition.parent.mkdir()
    detection.parent.mkdir()
    recognition.write_bytes(b"r")
    detection.write_bytes(b"d")

    resolved = face_models.resolve_model_paths(tmp_path)

    assert resolved.recognition == recognition
    assert resolved.detection == detection
    assert face_models.models_complete(tmp_path)


def test_resolves_standard_insightface_layout_and_nested_folder(tmp_path: Path):
    model_root = tmp_path / "antelopev2"
    model_root.mkdir()
    recognition = model_root / "glintr100.onnx"
    detection = model_root / "scrfd_10g_bnkps.onnx"
    recognition.write_bytes(b"r")
    detection.write_bytes(b"d")

    resolved = face_models.resolve_model_paths(tmp_path)

    assert resolved.recognition == recognition
    assert resolved.detection == detection
    assert face_models.models_complete(model_root)


def test_incomplete_model_folder_is_rejected(tmp_path: Path):
    (tmp_path / "glintr100.onnx").write_bytes(b"r")

    assert not face_models.models_complete(tmp_path)
    with pytest.raises(FileNotFoundError, match="standard InsightFace"):
        face_models.resolve_model_paths(tmp_path)
