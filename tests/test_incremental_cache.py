from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from backends import flux2, krea2, minimax_h3, wan
from musubi_tuner import cache_latents, cache_text_encoder_outputs
from musubi_tuner.dataset.cache_identity import cache_identity_matches, cache_recipe, identity_metadata, prepare_item_identity
from musubi_tuner.dataset.cache_io import save_latent_cache_common, save_text_encoder_output_cache_common
from musubi_tuner.dataset.image_video_dataset import ItemInfo


def _args(**overrides):
    values = {
        "validate_cache": True,
        "skip_existing": True,
        "num_workers": 1,
        "batch_size": 1,
        "keep_cache": True,
        "vae": "vae.safetensors",
        "cache_dtype": "float32",
        "teacher_conditions": None,
    }
    return Namespace(**(values | overrides))


def _item(tmp_path, name="portrait"):
    item = ItemInfo(name, "a person", (32, 32), (32, 32), content=np.zeros((32, 32, 3), dtype=np.uint8))
    item.latent_cache_path = str(tmp_path / (name + "_latents.safetensors"))
    item.text_encoder_output_cache_path = str(tmp_path / (name + "_text.safetensors"))
    return item


class _Dataset:
    def __init__(self, items):
        self.items = items
        self.datasource = items

    def retrieve_latent_cache_batches(self, *args, **kwargs):
        yield None, self.items

    def retrieve_text_encoder_output_cache_batches(self, *args):
        yield self.items

    def get_all_latent_cache_files(self):
        return [item.latent_cache_path for item in self.items if Path(item.latent_cache_path).exists()]

    def get_all_text_encoder_output_cache_files(self):
        return [item.text_encoder_output_cache_path for item in self.items if Path(item.text_encoder_output_cache_path).exists()]


@pytest.mark.parametrize("kind", ["latent", "text"])
def test_processing_missing_matching_stale_and_full_rebuild(tmp_path, kind):
    items = [_item(tmp_path), _item(tmp_path, "second")]
    dataset = _Dataset(items)
    encoded = []

    def encode(batch):
        for item in batch:
            encoded.append(item.item_key)
            if kind == "latent":
                save_latent_cache_common(item, {"latents": torch.zeros(1)}, "test")
            else:
                save_text_encoder_output_cache_common(item, {"embed": torch.zeros(1)}, "test", merge_existing=False)

    def run(args):
        if kind == "latent":
            cache_latents.encode_datasets([dataset], encode, args)
        else:
            files, paths = cache_text_encoder_outputs.prepare_cache_files_and_paths([dataset])
            cache_text_encoder_outputs.process_text_encoder_batches(
                1, args.skip_existing, 1, [dataset], files, paths, encode, cache_args=args
            )

    args = _args()
    run(args)
    assert encoded == ["portrait", "second"]
    encoded.clear()
    run(args)
    assert encoded == []
    if kind == "latent":
        items[0].content[0, 0, 0] = 1
    else:
        items[0].caption = "changed caption"
    run(args)
    assert encoded == ["portrait"]
    encoded.clear()
    run(_args(skip_existing=False))
    assert encoded == ["portrait", "second"]


@pytest.mark.parametrize("change", ["precision", "vae", "controls", "teacher", "dop"])
def test_identity_invalidates_changed_recipe_and_control_pixels(tmp_path, change):
    item = _item(tmp_path)
    args = _args()
    prepare_item_identity(item, cache_recipe(args, "latent"), "latent")
    save_file({"latents": torch.zeros(1)}, item.latent_cache_path, metadata=identity_metadata(item))
    assert cache_identity_matches(item, "latent")
    if change == "precision":
        args.cache_dtype = "bfloat16"
    if change == "vae":
        args.vae = "other_vae.safetensors"
    if change == "controls":
        item.control_content = [np.ones((4, 4, 3), dtype=np.uint8)]
    if change == "teacher":
        args.teacher_conditions = "subject_ref"
    if change == "dop":
        args.dop_class_word = "person"
    prepare_item_identity(item, cache_recipe(args, "latent"), "latent")
    assert not cache_identity_matches(item, "latent")


def test_checkpoint_replacement_and_legacy_cache_are_stale(tmp_path):
    model = tmp_path / "vae.safetensors"
    model.write_bytes(b"first")
    args = _args(vae=str(model))
    first = cache_recipe(args, "latent")
    model.write_bytes(b"replacement")
    assert cache_recipe(args, "latent") != first
    item = _item(tmp_path)
    prepare_item_identity(item, first, "latent")
    save_file({"latents": torch.zeros(1)}, item.latent_cache_path)
    assert not cache_identity_matches(item, "latent")
    Path(item.latent_cache_path).write_bytes(b"broken")
    assert not cache_identity_matches(item, "latent")


@pytest.mark.parametrize(
    "backend,extra",
    [
        (wan, {}),
        (flux2, {}),
        (krea2, {}),
        (minimax_h3, {}),
        (minimax_h3, {"minimax_h3_training_workflow": "Video · native"}),
    ],
)
def test_all_gui_backends_incremental_default_and_explicit_full_rebuild(backend, extra):
    settings = dict(
        dataset_config="data.toml",
        vae_model="vae",
        t5_model="t5",
        flux2_text_encoder="te",
        krea2_text_encoder="te",
        minimax_h3_text_encoder="te",
        minimax_h3_video_vae="vv",
        minimax_h3_audio_vae="av",
        recache_latents=True,
        recache_text=True,
        minimax_h3_guidance_distillation_protection=False,
        **extra,
    )
    commands = backend.build_cache_commands(settings, "python")
    assert len(commands) == 2
    assert all(command.count("--skip_existing") == 1 for command in commands)
    native = bool(extra)
    assert all(("--validate_cache" in command) != native for command in commands)
    settings.update(cache_incremental=False, dop_enabled=True, dop_cache_reuse=True)
    assert all("--skip_existing" not in command for command in backend.build_cache_commands(settings, "python"))


def test_provenance_composes_with_architecture_validator(tmp_path):
    item = _item(tmp_path)
    dataset = _Dataset([item])
    calls = []

    def encode(batch):
        calls.append(1)
        save_latent_cache_common(batch[0], {"latents": torch.zeros(1)}, "test")

    cache_latents.encode_datasets([dataset], encode, _args())
    cache_latents.encode_datasets([dataset], encode, _args(), skip_existing_validator=lambda path: False)
    assert calls == [1, 1]


def test_explicit_full_stage_caption_rebuild_overrides_incremental_policy():
    settings = {
        "dataset_config": "data.toml",
        "vae_model": "vae",
        "krea2_text_encoder": "te",
        "recache_latents": True,
        "recache_text": True,
        "cache_incremental": True,
        "use_staged_training": True,
        "staged_recache_text": True,
        "dop_enabled": True,
        "dop_cache_reuse": True,
    }
    latent, text = krea2.build_cache_commands(settings, "python")
    assert "--skip_existing" in latent
    assert "--skip_existing" not in text


def test_stale_text_rebuild_replaces_corrupt_file_and_drops_old_optional_keys(tmp_path):
    item = _item(tmp_path)
    prepare_item_identity(item, cache_recipe(_args(), "text"), "text")
    Path(item.text_encoder_output_cache_path).write_bytes(b"corrupt")
    save_text_encoder_output_cache_common(item, {"embed": torch.ones(1), "dop": torch.ones(1)}, "test")
    save_text_encoder_output_cache_common(item, {"embed": torch.zeros(1)}, "test")
    from safetensors import safe_open

    with safe_open(item.text_encoder_output_cache_path, framework="pt") as cache:
        assert list(cache.keys()) == ["embed"]
    assert cache_identity_matches(item, "text")
