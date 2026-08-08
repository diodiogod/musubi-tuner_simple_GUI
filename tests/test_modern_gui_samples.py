from pathlib import Path

import pytest

from modern_gui import samples
from modern_gui.samples import add_sample_source, discover_samples, find_nearby_sample_sources, remove_sample_source, resolve_sample_file


def test_sample_discovery_groups_epoch_variants(tmp_path: Path):
    run = tmp_path / "portrait"
    run.mkdir()
    (run / "portrait_e000001_00_20260724120000_42.png").write_bytes(b"one")
    (run / "portrait_e000002_00_20260724130000_42.png").write_bytes(b"two")
    (run / "notes.png").write_bytes(b"other")

    result = discover_samples(str(tmp_path), "portrait")

    assert result["count"] == 3
    assert len(result["groups"]) == 1
    assert [item["sequence"] for item in result["groups"][0]["items"]] == [2, 1]
    assert result["groups"][0]["items"][0]["sequence_label"] == "Epoch 2"
    assert result["groups"][0]["items"][0]["prefix"] == "portrait"
    assert result["groups"][0]["items"][0]["seed"] == "42"
    assert [item["name"] for item in result["ungrouped"]] == ["notes.png"]


def test_sample_discovery_includes_video_series(tmp_path: Path):
    run = tmp_path / "motion"
    run.mkdir()
    (run / "motion_e000001_00_20260724120000_42.mp4").write_bytes(b"one")
    (run / "motion_e000002_00_20260724130000_42.mp4").write_bytes(b"two")

    result = discover_samples(str(tmp_path), "motion")

    assert result["count"] == 2
    assert len(result["groups"]) == 1
    assert [item["sequence"] for item in result["groups"][0]["items"]] == [2, 1]
    assert all(item["media_kind"] == "video" for item in result["groups"][0]["items"])


def test_sample_discovery_merges_matching_prompt_across_version_folders(tmp_path: Path):
    current = tmp_path / "portrait-v2"
    previous = tmp_path / "portrait-v1"
    current.mkdir()
    previous.mkdir()
    (current / "portrait-v2_e000004_00_20260724130000_42.png").write_bytes(b"current")
    (previous / "portrait-v1_e000004_00_20260724120000_42.png").write_bytes(b"previous")
    (previous / "portrait-v1_e000004_01_20260724120000_42.png").write_bytes(b"other prompt")

    result = discover_samples(
        str(tmp_path),
        "portrait-v2",
        source_paths=[{"path": str(previous), "label": "Version 1"}],
    )

    assert len(result["groups"]) == 2
    matching = next(group for group in result["groups"] if len(group["items"]) == 2)
    assert {item["source_label"] for item in matching["items"]} == {"portrait-v2", "Version 1"}
    assert matching["prompt_index"] == 0


def test_sample_comparison_orders_matching_versions_by_last_modified(tmp_path: Path):
    current = tmp_path / "portrait-v2"
    previous = tmp_path / "portrait-v1"
    current.mkdir()
    previous.mkdir()
    newer = current / "portrait-v2_e000002_00_20260724120000_42.png"
    older = previous / "portrait-v1_e000009_00_20260724130000_42.png"
    newer.write_bytes(b"newer")
    older.write_bytes(b"older")
    import os

    os.utime(newer, (200, 200))
    os.utime(older, (100, 100))

    result = discover_samples(
        str(tmp_path),
        "portrait-v2",
        source_paths=[{"path": str(previous), "label": "Version 1"}],
    )

    matching = next(group for group in result["groups"] if len(group["items"]) == 2)
    assert matching["items"][0]["source_label"] == "portrait-v2"
    assert matching["items"][1]["source_label"] == "Version 1"


def test_sample_discovery_keeps_different_image_sizes_separate(tmp_path: Path):
    from PIL import Image

    current = tmp_path / "portrait-v2"
    previous = tmp_path / "portrait-v1"
    current.mkdir()
    previous.mkdir()
    Image.new("RGB", (768, 768)).save(current / "portrait-v2_e000004_00_20260724130000_42.png")
    Image.new("RGB", (1024, 1024)).save(previous / "portrait-v1_e000004_00_20260724120000_42.png")

    result = discover_samples(
        str(tmp_path),
        "portrait-v2",
        source_paths=[{"path": str(previous), "label": "Version 1"}],
    )

    assert len(result["groups"]) == 2
    assert {item["width"] for group in result["groups"] for item in group["items"]} == {768, 1024}


def test_sample_source_paths_are_persisted_and_removed(monkeypatch, tmp_path: Path):
    source_file = tmp_path / "sources.json"
    folder = tmp_path / "old-run"
    folder.mkdir()
    (folder / "old_e000001_00_20260724120000_7.png").write_bytes(b"sample")
    monkeypatch.setattr(samples, "SOURCES_PATH", source_file)

    added = add_sample_source(str(folder), "Old run")
    assert added == [{"path": str(folder.resolve()), "label": "Old run"}]
    assert samples.sample_source_status()[0]["count"] == 1
    assert remove_sample_source(str(folder)) == []


def test_nearby_sample_sources_use_current_output_parent(tmp_path: Path):
    current = tmp_path / "portrait-v2"
    previous = tmp_path / "portrait-v1"
    unrelated = tmp_path / "unrelated-empty"
    current.mkdir()
    previous.mkdir()
    unrelated.mkdir()
    (previous / "portrait-v1_e000001_00_20260724120000_7.png").write_bytes(b"sample")

    result = find_nearby_sample_sources(str(tmp_path), "portrait-v2")

    assert result == [{"path": str(previous.resolve()), "label": "portrait-v1", "count": 1}]


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


def test_sample_file_allows_video_inside_configured_output_root(tmp_path: Path):
    allowed = tmp_path / "output"
    allowed.mkdir()
    sample = allowed / "sample.mp4"
    sample.write_bytes(b"video")

    resolved, content_type = resolve_sample_file(str(sample), [allowed.resolve()])

    assert resolved == sample.resolve()
    assert content_type == "video/mp4"
