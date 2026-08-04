from contextlib import nullcontext

import torch
from safetensors.torch import load_file

from musubi_tuner.dataset.cache_io import (
    save_latent_cache_minimax_h3_image,
    save_text_encoder_output_cache_minimax_h3_image,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.minimax_h3_image_train_network import (
    MiniMaxH3ImageNetworkTrainer,
    minimax_h3_image_setup_parser,
)
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


def test_image_cache_contract_uses_shared_dataset_keys(tmp_path):
    item = ItemInfo("item", "caption", (32, 32), (32, 32))
    item.latent_cache_path = str(tmp_path / "item_0032x0032_mmh3.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "item_mmh3_te.safetensors")
    latent = torch.zeros(24, 1, 2, 2, dtype=torch.float16)
    hidden = torch.zeros(3, 5120, dtype=torch.bfloat16)

    save_latent_cache_minimax_h3_image(item, latent)
    save_text_encoder_output_cache_minimax_h3_image(item, hidden)
    latent_state = load_file(item.latent_cache_path)
    text_state = load_file(item.text_encoder_output_cache_path)

    assert set(latent_state) == {"latents_1x2x2_float16"}
    assert set(text_state) == {
        "varlen_mmh3_hidden_states_bfloat16",
        "varlen_mmh3_token_tags_int64",
    }
    assert torch.equal(text_state["varlen_mmh3_token_tags_int64"], torch.ones(3, dtype=torch.int64))


def test_minimax_h3_lora_modelspec_metadata():
    metadata = build_metadata(None, ARCHITECTURE_MINIMAX_H3, timestamp=0)

    assert metadata["modelspec.architecture"] == "MiniMax-H3/lora"
    assert metadata["modelspec.implementation"] == "https://huggingface.co/MiniMaxAI/MiniMax-H3"
