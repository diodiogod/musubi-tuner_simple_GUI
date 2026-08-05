from contextlib import nullcontext

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.dataset.cache_io import (
    save_latent_cache_minimax_h3_image,
    save_text_encoder_output_cache_minimax_h3_image,
    is_latent_cache_minimax_h3_image_current,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.minimax_h3_image_train_network import (
    MiniMaxH3ImageNetworkTrainer,
    minimax_h3_image_setup_parser,
    resolve_depth_vae_device,
)
from musubi_tuner.minimax_h3_image_cache_text_encoder_outputs import encode_and_save_batch, is_valid_minimax_h3_text_cache
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.utils.sai_model_spec import build_metadata


class _CPUAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def autocast():
        return nullcontext()


class _RecordingImageModel:
    def __init__(self):
        self.model_t = None

    def forward_image(self, noisy, model_t, text):
        self.model_t = model_t.detach().clone()
        assert text.shape == (1, 3, 5120)
        return torch.zeros_like(noisy)


class _FP32TextEncoder:
    @staticmethod
    def encode(_caption):
        return torch.ones(4, 5120, dtype=torch.float32), None


def test_trainer_image_flow_target_and_cleanness_conversion():
    trainer = MiniMaxH3ImageNetworkTrainer()
    model = _RecordingImageModel()
    latents = torch.ones(1, 24, 1, 2, 2, dtype=torch.bfloat16)
    noise = torch.full_like(latents, 0.25)
    output = trainer.call_dit(
        None,
        _CPUAccelerator(),
        model,
        latents,
        {"mmh3_hidden_states": [torch.zeros(3, 5120, dtype=torch.bfloat16)]},
        noise,
        torch.zeros_like(latents),
        torch.tensor([251.0]),
        torch.bfloat16,
    )
    torch.testing.assert_close(model.model_t, torch.tensor([0.75]))
    torch.testing.assert_close(output.target, latents - noise)


def test_experimental_parser_uses_24gb_safe_defaults():
    parser = minimax_h3_image_setup_parser(setup_parser_common())
    assert parser.get_default("network_module") == "musubi_tuner.networks.lora_minimax_h3"
    assert parser.get_default("blocks_to_swap") == 30
    assert parser.get_default("block_swap_h2d_only") is True
    assert parser.get_default("gradient_checkpointing") is True
    assert parser.get_default("timestep_sampling") == "krea2_shift"
    assert parser.get_default("depth_anchor_vae_device") == "training"
    assert parser.get_default("depth_anchor_every_n_steps") == 1


def test_secondary_depth_vae_device_selects_another_visible_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert resolve_depth_vae_device("secondary", torch.device("cuda:0")) == torch.device("cuda:1")
    assert resolve_depth_vae_device("secondary", torch.device("cuda:1")) == torch.device("cuda:0")


def test_secondary_depth_vae_device_requires_two_visible_gpus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="only one CUDA device"):
        resolve_depth_vae_device("secondary", torch.device("cuda:0"))


def test_image_cache_contract_uses_shared_dataset_keys(tmp_path):
    item = ItemInfo("item", "caption", (32, 32), (32, 32))
    item.latent_cache_path = str(tmp_path / "item_0032x0032_mmh3.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "item_mmh3_te.safetensors")
    latent = torch.zeros(24, 1, 2, 2, dtype=torch.float32)
    hidden = torch.zeros(3, 5120, dtype=torch.bfloat16)

    save_latent_cache_minimax_h3_image(item, latent)
    save_text_encoder_output_cache_minimax_h3_image(item, hidden)
    latent_state = load_file(item.latent_cache_path)
    text_state = load_file(item.text_encoder_output_cache_path)

    assert set(latent_state) == {"latents_1x2x2_float32"}
    with safe_open(item.latent_cache_path, framework="pt", device="cpu") as cache:
        assert cache.metadata()["posterior_policy"] == "video_vae=fp32"
    assert is_latent_cache_minimax_h3_image_current(item.latent_cache_path)
    assert set(text_state) == {
        "varlen_mmh3_hidden_states_bfloat16",
        "varlen_mmh3_token_tags_int64",
    }
    assert torch.equal(text_state["varlen_mmh3_token_tags_int64"], torch.ones(3, dtype=torch.int64))


def test_h3_text_cache_validator_accepts_requested_storage_precision(tmp_path):
    item = ItemInfo("item", "caption", (32, 32), (32, 32))
    item.text_encoder_output_cache_path = str(tmp_path / "item_mmh3_te.safetensors")

    save_text_encoder_output_cache_minimax_h3_image(item, torch.zeros(3, 5120, dtype=torch.bfloat16))
    assert is_valid_minimax_h3_text_cache(item, cache_dtype="bfloat16")
    assert not is_valid_minimax_h3_text_cache(item, cache_dtype="float32")

    save_text_encoder_output_cache_minimax_h3_image(item, torch.zeros(3, 5120, dtype=torch.float32))
    assert is_valid_minimax_h3_text_cache(item, cache_dtype="float32")
    assert not is_valid_minimax_h3_text_cache(item, cache_dtype="bfloat16")


def test_h3_text_cache_can_store_fp32_encoder_output_as_bf16(tmp_path):
    item = ItemInfo("item", "caption", (32, 32), (32, 32))
    item.text_encoder_output_cache_path = str(tmp_path / "item_mmh3_te.safetensors")

    encode_and_save_batch(_FP32TextEncoder(), [item], cache_dtype=torch.bfloat16)

    state = load_file(item.text_encoder_output_cache_path)
    assert state["varlen_mmh3_hidden_states_bfloat16"].dtype == torch.bfloat16


def test_h3_latent_cache_rejects_unmarked_precision(tmp_path):
    item = ItemInfo("item", "caption", (32, 32), (32, 32))
    item.latent_cache_path = str(tmp_path / "item_0032x0032_mmh3.safetensors")
    with pytest.raises(ValueError, match="FP32 VAE posterior"):
        save_latent_cache_minimax_h3_image(item, torch.zeros(24, 1, 2, 2, dtype=torch.float16))


def test_minimax_h3_lora_modelspec_metadata():
    metadata = build_metadata(None, ARCHITECTURE_MINIMAX_H3, timestamp=0)

    assert metadata["modelspec.architecture"] == "MiniMax-H3/lora"
    assert metadata["modelspec.implementation"] == "https://huggingface.co/MiniMaxAI/MiniMax-H3"
