from pathlib import Path

import pytest
from PIL import Image

from modern_gui.dataset_documents import (
    add_dataset,
    add_datasets,
    duplicate_dataset,
    load_document,
    inspect_dataset_sources,
    remove_dataset,
    save_document,
    split_dataset_subfolders,
    summarize_document,
    toggle_dataset_disabled,
    update_general,
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


def test_save_removes_empty_cache_directory_to_preserve_musubi_fallback(tmp_path: Path):
    destination = tmp_path / "dataset.toml"
    source = """[[datasets]]
image_directory = "images"
cache_directory = ""
resolution = 512
"""

    saved = save_document(str(destination), source)

    assert "cache_directory" not in saved["datasets"][0]["raw_values"]
    assert "cache_directory" not in destination.read_text(encoding="utf-8")


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


def test_add_dataset_uses_dropped_source_and_inherits_document_defaults():
    source = """[general]
resolution = [768, 1024]
num_repeats = 3
enable_bucket = false
caption_extension = ".txt"

[[datasets]]
image_directory = "existing"
"""

    added = add_dataset(
        source,
        "image",
        source_path="dataset.toml",
        folder_path=r"J:\\training\\images",
    )
    dataset = added["datasets"][-1]

    assert dataset["source"] == r"J:\\training\\images"
    assert dataset["resolution"] == [768, 1024]
    assert dataset["repeats"] == 3
    assert dataset["enable_bucket"] is False
    assert dataset["value_origins"]["resolution"] == "general"
    assert dataset["value_origins"]["num_repeats"] == "general"
    assert dataset["value_origins"]["enable_bucket"] == "general"
    assert dataset["value_origins"]["caption_extension"] == "general"
    assert dataset["cache_directory"] == ""
    assert "cache_directory" not in dataset["raw_values"]


def test_add_datasets_adds_all_dropped_folders():
    added = add_datasets(
        SAMPLE,
        "image",
        [r"J:\\training\\one", r"J:\\training\\two", r"J:\\training\\one"],
        source_path="dataset.toml",
    )

    assert [item["source"] for item in added["datasets"][-2:]] == [
        r"J:\\training\\one",
        r"J:\\training\\two",
    ]
    assert all("cache_directory" not in item["raw_values"] for item in added["datasets"][-2:])


def test_split_dataset_subfolders_copies_settings_and_replaces_empty_parent(tmp_path: Path):
    root = tmp_path / "car"
    low = root / "1_low_quality"
    high = root / "4_high_quality"
    low.mkdir(parents=True)
    high.mkdir()
    Image.new("RGB", (32, 24), "red").save(low / "low.png")
    Image.new("RGB", (32, 24), "blue").save(high / "high.png")
    (root / "empty").mkdir()
    source = f'''[general]
caption_extension = ".txt"

[[datasets]]
image_directory = "{str(root).replace(chr(92), "/")}"
cache_directory = "{str(tmp_path / "cache").replace(chr(92), "/")}"
resolution = [768, 768]
num_repeats = 3
'''

    split = split_dataset_subfolders(source, 0)

    assert [item["source"] for item in split["datasets"]] == [
        str(root / "1_low_quality").replace(chr(92), "/"),
        str(root / "4_high_quality").replace(chr(92), "/"),
    ]
    assert split["subfolder_scan"]["removed_parent"] is True
    assert all(item["repeats"] == 3 for item in split["datasets"])
    assert split["datasets"][0]["cache_directory"] == str(tmp_path / "cache" / "1_low_quality").replace(chr(92), "/")


def test_split_dataset_subfolders_preserves_parent_direct_media(tmp_path: Path):
    root = tmp_path / "car"
    root.mkdir()
    child = root / "high"
    child.mkdir()
    Image.new("RGB", (32, 24), "red").save(root / "direct.png")
    Image.new("RGB", (32, 24), "blue").save(child / "nested.png")
    source = f'''[[datasets]]
image_directory = "{str(root).replace(chr(92), "/")}"
resolution = 512
'''

    split = split_dataset_subfolders(source, 0)

    assert [item["source"] for item in split["datasets"]] == [
        str(root).replace(chr(92), "/"),
        str(child).replace(chr(92), "/"),
    ]
    assert split["subfolder_scan"]["removed_parent"] is False


def test_disabled_dataset_is_comment_marked_and_restores_all_settings():
    source = SAMPLE.replace('image_directory = "images"', 'image_directory = "images"\nfuture_option = "keep-me"')

    disabled = toggle_dataset_disabled(source, 0, True)

    assert disabled["datasets"] == []
    assert len(disabled["disabled_datasets"]) == 1
    assert disabled["disabled_datasets"][0]["source"] == "images"
    assert "# musubi-gui: disabled dataset v1" in disabled["text"]
    assert "# image_directory = \"images\"" in disabled["text"]
    assert 'image_directory = "images"' not in disabled["text"].split("# musubi-gui: disabled dataset v1", 1)[0]

    restored = toggle_dataset_disabled(disabled["text"], 0, False)

    assert restored["datasets"][0]["source"] == "images"
    assert restored["datasets"][0]["raw_values"]["future_option"] == "keep-me"
    assert restored["disabled_datasets"] == []
    assert "musubi-gui: disabled dataset" not in restored["text"]


def test_visual_updates_preserve_disabled_dataset_blocks():
    source = toggle_dataset_disabled(SAMPLE, 0, True)["text"]

    updated = update_general(source, {"batch_size": 2})

    assert updated["datasets"] == []
    assert updated["general"]["batch_size"] == 2
    assert updated["disabled_datasets"][0]["resolution"] == [1024, 1024]
    assert "# musubi-gui: disabled dataset v1" in updated["text"]


def test_save_and_load_round_trip_keeps_disabled_sources(tmp_path: Path):
    destination = tmp_path / "dataset.toml"
    source = toggle_dataset_disabled(SAMPLE, 0, True)["text"]

    save_document(str(destination), source)
    loaded = load_document(str(destination))

    assert loaded["datasets"] == []
    assert loaded["disabled_datasets"][0]["source"] == "images"
    assert "# musubi-gui: disabled dataset v1" in destination.read_text(encoding="utf-8")


def test_disabled_source_keeps_its_visual_position_and_restores_in_place():
    source = SAMPLE + '''
[[datasets]]
image_directory = "second"
resolution = 512

[[datasets]]
image_directory = "third"
resolution = 512
'''

    disabled = toggle_dataset_disabled(source, 1, True, position=1)

    assert [item["source"] for item in disabled["datasets"]] == ["images", "third"]
    assert disabled["disabled_datasets"][0]["source"] == "second"
    assert disabled["disabled_datasets"][0]["position"] == 1
    assert "disabled dataset v1 position=1" in disabled["text"]

    restored = toggle_dataset_disabled(disabled["text"], 0, False, position=1)

    assert [item["source"] for item in restored["datasets"]] == ["images", "second", "third"]


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
