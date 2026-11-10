"""Mask sidecars never become photos, unhealthy gallery items or stale cached samples."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from musubi_tuner.dataset.mask_files import is_mask_image
from modern_gui.dataset_documents import split_dataset_subfolders
from modern_gui.dataset_media import audit_dataset_sources, list_dataset_media


def photo(folder, name, caption=True):
    path = folder / name
    Image.new("RGB", (32, 32), "red").save(path)
    if caption:
        path.with_suffix(".txt").write_text("a photo", encoding="utf-8")
    return path


def document(folder):
    return f'[[datasets]]\nimage_directory = "{folder.as_posix()}"\nresolution = [32, 32]\nnum_repeats = 3\ncaption_extension = ".txt"\n'


@pytest.mark.parametrize("name,expected", [
    ("person-masklabel.png", True), ("person-MASKLABEL.WEBP", True),
    ("person-masklabel.jpg", True), ("person_with_mask.png", False),
    ("mask.png", False), ("masklabel.png", False),
    ("person-masklabel-example.png", False), ("person_mask.png", False),
])
def test_only_the_reserved_mask_sidecar_suffix_is_excluded(name, expected):
    assert is_mask_image(Path(name)) is expected


def test_gallery_audit_and_cache_discovery_ignore_masks_even_with_captions(tmp_path):
    from musubi_tuner.dataset.media_utils import glob_images
    from musubi_tuner.dataset.datasources import ImageDirectoryDatasource
    photo(tmp_path, "person.png")
    photo(tmp_path, "person_with_mask.jpg")
    photo(tmp_path, "person-masklabel.png", caption=False)
    photo(tmp_path, "captioned-MASKLABEL.PNG")
    # A broken mask must not turn the source into an unreadable-media warning.
    (tmp_path / "broken-masklabel.png").write_bytes(b"not an image")
    nested = tmp_path / "masks"
    nested.mkdir()
    photo(nested, "nested-masklabel.png", caption=False)
    source = document(tmp_path)
    payload = list_dataset_media(source, 0)
    overview = payload["overview"]
    assert overview["media_count"] == overview["trainer_usable_count"] == 2
    assert overview["caption_count"] == 2
    assert overview["missing_caption_count"] == overview["empty_caption_count"] == 0
    assert {item["name"] for item in payload["items"]} == {"person.png", "person_with_mask.jpg"}
    audit = audit_dataset_sources(source)["datasets"][0]
    assert audit["effective_samples"] == 6
    assert audit["ignored_nested_count"] == audit["unreadable_count"] == 0
    assert {Path(p).name for p in glob_images(str(tmp_path))} == {"person.png", "person_with_mask.jpg"}
    assert len(ImageDirectoryDatasource(str(tmp_path), caption_extension=".txt")) == 2
    assert (tmp_path / "broken-masklabel.png").is_file()  # ignore, never delete


def test_folder_expansion_does_not_add_mask_only_sources(tmp_path):
    photo(tmp_path, "root-masklabel.png", caption=False)
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    photo(images, "real.png")
    photo(masks, "real-masklabel.png", caption=False)
    result = split_dataset_subfolders(document(tmp_path), 0)
    assert len(result["datasets"]) == 1
    assert Path(result["datasets"][0]["source"]) == images
    assert result["subfolder_scan"]["removed_parent"] is True


def test_classic_source_inspection_reports_only_photos(tmp_path):
    from dataset_config_builder import DatasetConfigBuilder
    photo(tmp_path, "real.png")
    photo(tmp_path, "real-masklabel.png", caption=False)
    nested = tmp_path / "masks"
    nested.mkdir()
    photo(nested, "nested-masklabel.png", caption=False)
    def field(value):
        return SimpleNamespace(get=lambda: value)
    label = {}
    builder = SimpleNamespace(
        dataset_vars={"source": field(str(tmp_path)), "caption_extension": field(".txt"), "num_repeats": field("3")},
        general_vars={"caption_extension": field(".txt")}, datasets=[{"kind": "image"}], selected_index=0,
        inspect_label=SimpleNamespace(configure=lambda **kwargs: label.update(kwargs)),
    )
    DatasetConfigBuilder._inspect_selected_source(builder)
    assert label["text"] == "1 media · 1 captions · 3 effective samples"


def test_preexisting_mask_latents_are_excluded_from_training(tmp_path):
    from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo
    from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE
    from musubi_tuner.moss_image.cache import save_embeddings, save_latents
    image = photo(tmp_path, "person.png")
    mask = photo(tmp_path, "person-masklabel.png")
    dataset = ImageDataset(resolution=(32, 32), caption_extension=".txt", batch_size=1,
        num_repeats=2, enable_bucket=False, bucket_no_upscale=False,
        image_directory=str(tmp_path), cache_directory=str(tmp_path), architecture=ARCHITECTURE_MOSS_IMAGE)
    for path, value in ((image, 1), (mask, 99)):
        item = ItemInfo(str(path), "a photo", (32, 32))
        item.latent_cache_path = dataset.get_latent_cache_path(item)
        item.text_encoder_output_cache_path = dataset.get_text_encoder_output_cache_path(item)
        save_latents(item, torch.full((128, 1, 2, 2), float(value)))
        save_embeddings(item, torch.ones(3, 4096))
    dataset.prepare_for_training()
    dataset.set_seed(42, SimpleNamespace(value=0))
    assert dataset.num_train_items == 2  # one real photo, repeated twice
    for index in range(len(dataset)):
        assert torch.equal(dataset[index]["latents"], torch.ones(1, 128, 1, 2, 2))
