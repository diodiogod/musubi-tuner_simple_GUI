from pathlib import Path

import pytest

from modern_gui.samples import discover_samples, resolve_sample_file


def test_sample_discovery_groups_epoch_variants(tmp_path: Path):
    run = tmp_path / "portrait"
    run.mkdir()
    (run / "portrait_e000001_00_20260724120000_42.png").write_bytes(b"one")
    (run / "portrait_e000002_00_20260724130000_42.png").write_bytes(b"two")
    (run / "notes.png").write_bytes(b"other")

    result = discover_samples(str(tmp_path), "portrait")

    assert result["count"] == 3
    assert len(result["groups"]) == 1
    assert [item["sequence"] for item in result["groups"][0]["items"]] == [1, 2]
    assert result["groups"][0]["items"][0]["sequence_label"] == "Epoch 1"
    assert result["groups"][0]["items"][0]["prefix"] == "portrait"
    assert result["groups"][0]["items"][0]["seed"] == "42"
    assert [item["name"] for item in result["ungrouped"]] == ["notes.png"]


def test_sample_file_is_restricted_to_configured_output_roots(tmp_path: Path):
    allowed = tmp_path / "output"
    allowed.mkdir()
    sample = allowed / "sample.webp"
    sample.write_bytes(b"image")
    outside = tmp_path / "private.png"
    outside.write_bytes(b"private")

    resolved, content_type = resolve_sample_file(str(sample), [allowed.resolve()])

    assert resolved == sample.resolve()
    assert content_type == "image/webp"
    with pytest.raises(PermissionError):
        resolve_sample_file(str(outside), [allowed.resolve()])
