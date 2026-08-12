from pathlib import Path

import pytest
from PIL import Image

from modern_gui.dataset_documents import (
    add_dataset,
    duplicate_dataset,
    load_document,
    inspect_dataset_sources,
    remove_dataset,
    save_document,
    summarize_document,
    update_dataset,
)


SAMPLE = """# This comment must survive web-editor round trips.
[general]
caption_extension = ".txt"
batch_size = 1

[[datasets]]
image_directory = "images"
cache_directory = "cache"
resolution = [1024, 1024]
num_repeats = 4
enable_bucket = true
"""


def test_summary_exposes_visual_dataset_information_without_changing_source():
    summary = summarize_document(SAMPLE, "dataset.toml")

    dataset = summary["datasets"][0]
    assert dataset["kind"] == "image"
    assert dataset["source"] == "images"
    assert dataset["resolution"] == [1024, 1024]
    assert dataset["repeats"] == 4
    assert dataset["batch_size"] == 1
    assert dataset["cache_directory"] == "cache"
    assert dataset["caption_extension"] == ".txt"
    assert "# This comment must survive" in summary["text"]


def test_save_is_parse_checked_and_preserves_comments(tmp_path: Path):
    destination = tmp_path / "dataset.toml"

    saved = save_document(str(destination), SAMPLE)
    loaded = load_document(str(destination))

    assert saved["path"] == str(destination.resolve())
    assert loaded["datasets"][0]["resolution"] == [1024, 1024]
    assert "# This comment must survive" in destination.read_text(encoding="utf-8")
    assert not destination.with_suffix(".toml.tmp").exists()


def test_invalid_toml_does_not_replace_existing_file(tmp_path: Path):
    destination = tmp_path / "dataset.toml"
    destination.write_text(SAMPLE, encoding="utf-8")

    with pytest.raises(Exception):
        save_document(str(destination), "[[datasets]\nbroken = true")

    assert destination.read_text(encoding="utf-8") == SAMPLE


def test_missing_source_and_resolution_are_reported():
    summary = summarize_document("[[datasets]]\nnum_repeats = 1\n")

    messages = [issue["message"] for issue in summary["issues"]]
    assert "Choose a image directory or JSONL file." in messages
    assert "Resolution is required." in messages


def test_visual_update_preserves_comments_and_unknown_fields():
    source = SAMPLE.replace("enable_bucket = true", 'enable_bucket = true\nfuture_option = "keep-me"')

    updated = update_dataset(source, 0, {"resolution": [768, 1024], "num_repeats": 7})

    assert updated["datasets"][0]["resolution"] == [768, 1024]
    assert updated["datasets"][0]["repeats"] == 7
    assert 'future_option = "keep-me"' in updated["text"]
    assert "# This comment must survive" in updated["text"]


def test_add_duplicate_and_remove_dataset_round_trip():
    added = add_dataset(SAMPLE, "video")
    assert [item["kind"] for item in added["datasets"]] == ["image", "video"]

    duplicated = duplicate_dataset(added["text"], 0)
    assert [item["kind"] for item in duplicated["datasets"]] == ["image", "image", "video"]

    removed = remove_dataset(duplicated["text"], 1)
    assert [item["kind"] for item in removed["datasets"]] == ["image", "video"]


def test_add_h3_video_dataset_uses_released_frame_geometry():
    added = add_dataset(SAMPLE, "video", architecture="minimax_h3")

    assert added["datasets"][-1]["target_frames"] == [124]


def test_source_inspection_reports_media_captions_and_resolutions(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (640, 480), "red").save(image_dir / "captioned.png")
    Image.new("RGB", (1024, 1024), "blue").save(image_dir / "missing.jpg")
    (image_dir / "captioned.txt").write_text("a caption", encoding="utf-8")
    source = f"""[[datasets]]
image_directory = {str(image_dir)!r}
resolution = [1024, 1024]
caption_extension = ".txt"
"""

    report = inspect_dataset_sources(source)["datasets"][0]

    assert report["media_count"] == 2
    assert report["caption_count"] == 1
    assert report["missing_caption_count"] == 1
    assert report["resolutions"] == {"640×480": 1, "1024×1024": 1}
