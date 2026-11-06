from types import SimpleNamespace

import numpy as np
import torch

from musubi_tuner.moss_image import cache
from musubi_tuner import moss_image_cache_latents as latent_cache


def test_latent_encoding_keeps_packed_channels(monkeypatch):
    saved = []

    class VAE:
        device = torch.device("cpu")
        dtype = torch.float32

        def encode(self, pixels):
            assert pixels.shape == (1, 3, 32, 48)
            assert pixels.min() == -1 and pixels.max() == -1
            return torch.ones(1, 128, 2, 3)

    monkeypatch.setattr(latent_cache, "save_latents", lambda item, latent: saved.append(latent))
    latent_cache.encode_and_save_batch(VAE(), [SimpleNamespace(content=np.zeros((32, 48, 3), dtype=np.uint8))])
    assert saved[0].shape == (128, 1, 2, 3)


def test_cache_keys_are_separate_from_flux_and_krea(monkeypatch):
    saved = []
    monkeypatch.setattr(cache, "save_latent_cache_common", lambda item, tensors, arch: saved.append((tensors, arch)))
    monkeypatch.setattr(cache, "save_text_encoder_output_cache_common", lambda item, tensors, arch: saved.append((tensors, arch)))
    cache.save_latents(None, torch.zeros(128, 1, 2, 3))
    cache.save_embeddings(None, torch.zeros(3, 4096, dtype=torch.bfloat16),
                          torch.zeros(2, 4096, dtype=torch.bfloat16), torch.tensor([1]))
    assert all(arch == "moss_image" for _, arch in saved)
    assert "latents_1x2x3_float32" in saved[0][0]
    assert "varlen_moss_embed_bfloat16" in saved[1][0]
    assert "varlen_dop_moss_embed_bfloat16" in saved[1][0]
    assert "dop_signature" in saved[1][0]


def test_cache_files_round_trip_through_real_dataset_loader(tmp_path):
    from PIL import Image
    from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo
    from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE
    from musubi_tuner.training.dop import dop_signature

    image_path = tmp_path / "portrait.png"
    Image.new("RGB", (32, 32)).save(image_path)
    dataset = ImageDataset(resolution=(32, 32), caption_extension=".txt", batch_size=1,
                           num_repeats=1, enable_bucket=False, bucket_no_upscale=False,
                           image_directory=str(tmp_path), cache_directory=str(tmp_path),
                           architecture=ARCHITECTURE_MOSS_IMAGE)
    item = ItemInfo(str(image_path), "a person", (32, 32))
    item.latent_cache_path = dataset.get_latent_cache_path(item)
    item.text_encoder_output_cache_path = dataset.get_text_encoder_output_cache_path(item)
    cache.save_latents(item, torch.ones(128, 1, 2, 2))
    cache.save_embeddings(item, torch.ones(3, 4096, dtype=torch.bfloat16),
                          torch.ones(2, 4096, dtype=torch.bfloat16), dop_signature("token", "person"))
    dataset.prepare_for_training()
    dataset.set_seed(42, SimpleNamespace(value=0))
    batch = dataset[0]
    assert batch["latents"].shape == (1, 128, 1, 2, 2)
    assert batch["moss_embed"][0].shape == (3, 4096)
    assert batch["dop_moss_embed"][0].shape == (2, 4096)
    assert "dop_signature" in batch
