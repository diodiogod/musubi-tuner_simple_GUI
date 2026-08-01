import json
from pathlib import Path

import pytest
from PIL import Image

from modern_gui.dataset_documents import (
    DocumentConflictError,
    duplicate_dataset,
    move_dataset,
    save_document,
    summarize_document,
    update_dataset,
    update_general,
)
from modern_gui.dataset_media import (
    CaptionConflictError,
    MediaTokenError,
    dataset_source_location,
    list_dataset_media,
    resolve_media_token,
    save_media_caption,
)


def _image(path: Path, size=(64, 48), color="red"):
    Image.new("RGB", size, color).save(path)


def _directory_toml(directory: Path, *, repeats=2):
    return f"""[general]
caption_extension = ".txt"
batch_size = 3

[[datasets]]
image_directory = {str(directory)!r}
resolution = 512
num_repeats = {repeats}
future_key = "preserve"
"""


def test_summary_separates_raw_values_from_effective_inheritance():
    source = """[general]
resolution = 768
num_repeats = 4
batch_size = 2
caption_extension = ".caption.txt"

[[datasets]]
image_directory = "images"
"""

    summary = summarize_document(source)
    dataset = summary["datasets"][0]

    assert dataset["raw_values"] == {"image_directory": "images"}
    assert dataset["effective_values"]["resolution"] == 768
    assert dataset["effective_values"]["num_repeats"] == 4
    assert dataset["value_origins"]["batch_size"] == "general"
    assert set(dataset["inherited_from_general"]) >= {
        "resolution",
        "num_repeats",
        "batch_size",
        "caption_extension",
    }
    assert not any(issue["message"] == "Resolution is required." for issue in summary["issues"])


def test_visual_noop_keeps_scalar_resolution_and_inherited_fields_absent():
    source = """# keep
[general]
batch_size = 2
caption_extension = ".txt"

[[datasets]]
image_directory = "images"
resolution = 512
num_repeats = 3
"""

    updated = update_dataset(
        source,
        0,
        {
            "image_directory": "images",
            "image_jsonl_file": "",
            "resolution": 512,
            "num_repeats": 3,
            "batch_size": "",
            "caption_extension": "",
        },
    )

    assert updated["text"] == source
    assert updated["datasets"][0]["raw_values"]["resolution"] == 512
    assert "batch_size" not in updated["datasets"][0]["raw_values"]
    assert "caption_extension" not in updated["datasets"][0]["raw_values"]


def test_duplicate_clears_cache_and_reorder_preserves_unknown_fields():
    source = """[[datasets]]
image_directory = "one"
cache_directory = "cache-one"
resolution = [512, 512]
future_key = "one"

[[datasets]]
image_directory = "two"
cache_directory = "cache-two"
resolution = [768, 768]
future_key = "two"
"""

    duplicated = duplicate_dataset(source, 0)
    assert "cache_directory" not in duplicated["datasets"][1]["raw_values"]
    assert duplicated["datasets"][1]["raw_values"]["future_key"] == "one"

    moved = move_dataset(duplicated["text"], 2, 0)
    assert [item["raw_values"]["future_key"] for item in moved["datasets"]] == ["two", "one", "one"]

    fallback_duplicate = duplicate_dataset(
        '[[datasets]]\nimage_directory = "same"\nresolution = 512\n',
        0,
    )
    assert any(
        "Effective cache location is already used" in issue["message"]
        for issue in fallback_duplicate["issues"]
    )


def test_general_defaults_and_source_format_switch_are_lossless():
    source = """# document note
[general]
batch_size = 1 # keep inline

[[datasets]]
image_directory = "images"
resolution = 512
future_key = "keep"
"""

    defaults = update_general(
        source,
        {"batch_size": 2, "caption_extension": ".caption.txt"},
    )
    switched = update_dataset(
        defaults["text"],
        0,
        {"image_directory": "", "image_jsonl_file": "manifest.jsonl"},
    )

    assert "# document note" in switched["text"]
    assert "batch_size = 2 # keep inline" in switched["text"]
    assert switched["general"]["caption_extension"] == ".caption.txt"
    assert switched["datasets"][0]["source_mode"] == "jsonl"
    assert "image_directory" not in switched["datasets"][0]["raw_values"]
    assert switched["datasets"][0]["raw_values"]["future_key"] == "keep"


def test_media_inventory_matches_musubi_direct_child_scan(tmp_path: Path):
    root = tmp_path / "images"
    nested = root / "nested"
    root.mkdir()
    nested.mkdir()
    _image(root / "ready.png")
    (root / "ready.txt").write_text("caption", encoding="utf-8")
    _image(root / "unsupported.gif")
    _image(nested / "ignored.png")
    (nested / "ignored.txt").write_text("nested", encoding="utf-8")

    payload = list_dataset_media(_directory_toml(root))

    assert payload["overview"]["media_count"] == 1
    assert payload["overview"]["trainer_usable_count"] == 1
    assert payload["overview"]["effective_samples"] == 2
    assert [item["name"] for item in payload["items"]] == ["ready.png"]
    assert dataset_source_location(_directory_toml(root), 0) == root.resolve()


def test_missing_image_caption_is_excluded_and_can_be_created_atomically(tmp_path: Path):
    root = tmp_path / "images"
    root.mkdir()
    _image(root / "missing.png")
    payload = list_dataset_media(_directory_toml(root))
    item = payload["items"][0]

    assert item["caption_state"] == "missing"
    assert item["training_state"] == "excluded"
    assert resolve_media_token(item["token"]).path == (root / "missing.png").resolve()

    saved = save_media_caption(item["token"], "new caption", item["caption_revision"])
    assert saved["caption_state"] == "present"
    assert (root / "missing.txt").read_text(encoding="utf-8") == "new caption"

    with pytest.raises(CaptionConflictError):
        save_media_caption(item["token"], "stale edit", item["caption_revision"])
    with pytest.raises(MediaTokenError):
        resolve_media_token("not-a-real-token")


def test_layer_targets_and_controls_are_paired_without_false_missing_captions(tmp_path: Path):
    root = tmp_path / "images"
    controls = tmp_path / "controls"
    root.mkdir()
    controls.mkdir()
    _image(root / "subject.png")
    _image(root / "subject_0.png", color="green")
    (root / "subject.txt").write_text("layered subject", encoding="utf-8")
    _image(controls / "subject.png", color="blue")
    source = f"""[general]
caption_extension = ".txt"

[[datasets]]
image_directory = {str(root)!r}
control_directory = {str(controls)!r}
resolution = 512
multiple_target = true
"""

    payload = list_dataset_media(source)
    by_name = {item["name"]: item for item in payload["items"]}

    assert payload["overview"]["primary_count"] == 1
    assert payload["overview"]["missing_caption_count"] == 0
    assert by_name["subject.png"]["training_state"] == "eligible"
    assert len(by_name["subject.png"]["controls"]) == 1
    assert by_name["subject_0.png"]["role"] == "target"
    assert by_name["subject_0.png"]["training_state"] == "paired_target"


def test_jsonl_inventory_and_caption_edit_preserve_other_lines(tmp_path: Path):
    image = tmp_path / "item.png"
    _image(image)
    manifest = tmp_path / "items.jsonl"
    second_line = '{"image_path":"untouched.png","caption":"leave me", "future": 3}\r\n'
    manifest.write_text(
        json.dumps({"image_path": str(image), "caption": "old", "future": {"nested": True}})
        + "\r\n"
        + second_line,
        encoding="utf-8",
        newline="",
    )
    source = f"""[[datasets]]
image_jsonl_file = {str(manifest)!r}
cache_directory = {str(tmp_path / "cache")!r}
resolution = [512, 512]
"""

    payload = list_dataset_media(source)
    item = payload["items"][0]
    assert payload["source"]["mode"] == "jsonl"
    assert item["caption"] == "old"

    save_media_caption(item["token"], "updated inline caption", item["caption_revision"])
    lines = manifest.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])

    assert first["caption"] == "updated inline caption"
    assert first["future"] == {"nested": True}
    assert lines[1] == second_line.rstrip("\r\n")


def test_disk_revision_blocks_overwriting_external_toml_change(tmp_path: Path):
    destination = tmp_path / "dataset.toml"
    source = _directory_toml(tmp_path / "images")
    loaded = save_document(str(destination), source)
    destination.write_text(source + "\n# external edit\n", encoding="utf-8")

    with pytest.raises(DocumentConflictError):
        save_document(str(destination), source, loaded["disk_revision"])


def test_unknown_toml_dates_are_json_safe():
    summary = summarize_document(
        """[[datasets]]
image_directory = "images"
resolution = 512
future_date = 2026-07-28
"""
    )

    assert summary["datasets"][0]["raw_values"]["future_date"] == "2026-07-28"
    assert "future_date = 2026-07-28" in summary["text"]
