from contextlib import nullcontext
from types import SimpleNamespace

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
    DiTOutput,
    MiniMaxH3ImageNetworkTrainer,
    minimax_h3_image_setup_parser,
    resolve_depth_vae_device,
)
from musubi_tuner.minimax_h3_image_cache_text_encoder_outputs import encode_and_save_batch, is_valid_minimax_h3_text_cache
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.utils.sai_model_spec import build_metadata
from musubi_tuner.training.h3_guidance_protection import build_guided_target
import musubi_tuner.minimax_h3_image_train_network as h3_training


class _CPUAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def unwrap_model(model):
        return model


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


class _PreviewTransformer:
    def park_resident_block_weights_for_decode(self, _minimum_free):
        return []

    def restore_parked_block_weights(self, _parked):
        return None


class _PreviewVAE:
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


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
    assert parser.get_default("h3_guidance_distillation_protection") is False
    assert parser.get_default("h3_guidance_distillation_scale") == 4.0
    assert parser.get_default("h3_guidance_distillation_schedule") == "sigma"


def test_h3_guidance_protection_amplifies_target_away_from_unconditional_prediction():
    unconditional = torch.tensor([1.0, 2.0])
    normal_target = torch.tensor([3.0, -1.0])

    protected = build_guided_target(unconditional, normal_target, 3.0, schedule="constant")

    torch.testing.assert_close(protected, torch.tensor([7.0, -7.0]))
    assert not protected.requires_grad


def test_h3_process_batch_uses_empty_prompt_pass_before_protected_primary_target(monkeypatch):
    trainer = MiniMaxH3ImageNetworkTrainer()
    calls = []
    captured = {}
    latents = torch.zeros(1, 24, 1, 2, 2)
    noise = torch.zeros_like(latents)
    normal_target = torch.full_like(latents, 3.0)

    monkeypatch.setattr(
        trainer,
        "get_noisy_model_input_and_timesteps",
        lambda *args, **kwargs: (torch.zeros_like(latents), torch.tensor([500.0])),
    )

    def fake_call(_args, _accelerator, _transformer, _latents, batch, *_rest, **_kwargs):
        calls.append(batch["mmh3_hidden_states"])
        if len(calls) == 1:
            return DiTOutput(pred=torch.ones_like(latents), target=normal_target)
        return DiTOutput(pred=torch.zeros_like(latents, requires_grad=True), target=normal_target)

    def fake_compute(_args, output, *_rest, **_kwargs):
        captured["target"] = output.target
        return torch.nn.functional.mse_loss(output.pred, output.target), {}

    monkeypatch.setattr(trainer, "call_dit", fake_call)
    monkeypatch.setattr(trainer, "compute_loss", fake_compute)
    args = SimpleNamespace(
        h3_guidance_distillation_protection=True,
        h3_guidance_distillation_scale=4.0,
        h3_guidance_distillation_schedule="sigma",
        depth_anchor_weight=0.0,
        depth_anchor_every_n_steps=1,
        dop_loss_weight=0.0,
    )
    caption = [torch.full((2, 5120), 2.0)]
    unconditional = [torch.zeros(1, 5120)]

    loss, metrics = trainer.process_batch(
        args,
        _CPUAccelerator(),
        object(),
        object(),
        {"timesteps": torch.tensor([0.5]), "mmh3_hidden_states": caption,
         "mmh3_unconditional_hidden_states": unconditional},
        latents,
        noise,
        object(),
        torch.bfloat16,
        torch.bfloat16,
        None,
        0,
    )

    assert calls == [unconditional, caption]
    torch.testing.assert_close(captured["target"], torch.full_like(latents, 6.0))
    assert loss.item() == pytest.approx(36.0)
    assert metrics["loss/h3_guidance_target_delta"].item() == pytest.approx(9.0)


def test_h3_sigma_schedule_fades_protection_toward_clean_timesteps():
    unconditional = torch.zeros(2, 1)
    target = torch.ones(2, 1)
    protected = build_guided_target(
        unconditional, target, 4.0, sigma=torch.tensor([1.0, 0.0]), schedule="sigma"
    )
    torch.testing.assert_close(protected, torch.tensor([[4.0], [1.0]]))


def test_hybrid_protection_bypasses_only_user_lora_for_sparse_reference(monkeypatch):
    trainer = MiniMaxH3ImageNetworkTrainer()
    latents = torch.zeros(1, 24, 1, 2, 2)
    multipliers = []

    class Network:
        multiplier = 1.0

        def set_multiplier(self, value):
            self.multiplier = value

    network = Network()
    monkeypatch.setattr(
        trainer,
        "get_noisy_model_input_and_timesteps",
        lambda *args, **kwargs: (torch.zeros_like(latents), torch.tensor([500.0])),
    )

    def fake_call(*_args, **_kwargs):
        multipliers.append(network.multiplier)
        prediction = torch.full_like(latents, network.multiplier, requires_grad=network.multiplier != 0)
        return DiTOutput(pred=prediction, target=torch.zeros_like(latents))

    monkeypatch.setattr(trainer, "call_dit", fake_call)
    monkeypatch.setattr(
        trainer,
        "compute_loss",
        lambda _args, output, *_rest, **_kwargs: (output.pred.square().mean(), {}),
    )
    args = SimpleNamespace(
        h3_quality_protection_method="assistant_preservation",
        h3_base_preservation_enabled=True,
        h3_base_preservation_reference="assistant",
        h3_base_preservation_loss_weight=0.05,
        h3_base_preservation_every_n_steps=10,
        h3_guidance_distillation_protection=False,
        depth_anchor_weight=0.0,
        depth_anchor_every_n_steps=1,
        dop_loss_weight=0.0,
    )

    loss, metrics = trainer.process_batch(
        args, _CPUAccelerator(), object(), network,
        {"timesteps": torch.tensor([0.5]), "mmh3_hidden_states": [torch.zeros(1, 5120)]},
        latents, torch.zeros_like(latents), object(), torch.bfloat16, torch.bfloat16, None, 9,
    )

    assert multipliers == [0.0, 1.0]
    assert network.multiplier == 1.0
    assert loss.item() == pytest.approx(1.05)
    assert metrics["loss/h3_base_preservation"].item() == pytest.approx(1.0)


def test_five_frame_training_preview_uses_video_sampler_and_returns_video_grid(monkeypatch):
    calls = {}

    def fake_sample_video(_transformer, _hidden, **kwargs):
        calls["sample"] = kwargs
        return torch.zeros(1, 24, 2, 2, 2)

    def fake_decode_video(_vae, latent, *, frame_count):
        calls["decode"] = (tuple(latent.shape), frame_count)
        return torch.full((5, 4, 6, 3), 128, dtype=torch.uint8)

    monkeypatch.setattr(h3_training, "sample_video_latent", fake_sample_video)
    monkeypatch.setattr(h3_training, "decode_video_latent", fake_decode_video)
    monkeypatch.setattr(h3_training, "clean_memory_on_device", lambda _device: None)
    trainer = MiniMaxH3ImageNetworkTrainer()
    args = SimpleNamespace(
        minimax_h3_preview_decode_min_free_gb=0,
        depth_anchor_weight=0,
        keep_depth_vae_on_device=False,
    )

    pixels = trainer.do_inference(
        _CPUAccelerator(), args, {"mmh3_hidden_states": torch.zeros(3, 5120), "seed": 7},
        _PreviewVAE(), torch.bfloat16, _PreviewTransformer(), 12.0, 4, 64, 64, 5,
        None, False, 1.0, 1.0,
    )

    assert calls["sample"]["frame_count"] == 5
    assert calls["decode"] == ((1, 24, 2, 2, 2), 5)
    assert pixels.shape == (1, 3, 5, 4, 6)
    assert pixels.dtype == torch.float32
    torch.testing.assert_close(pixels.mean(), torch.tensor(128 / 255))


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


def test_h3_text_cache_validator_requires_empty_prompt_when_quality_protection_is_enabled(tmp_path):
    item = ItemInfo("item", "caption", (32, 32), (32, 32))
    item.text_encoder_output_cache_path = str(tmp_path / "item_mmh3_te.safetensors")
    caption = torch.zeros(3, 5120, dtype=torch.bfloat16)
    unconditional = torch.ones(2, 5120, dtype=torch.bfloat16)

    save_text_encoder_output_cache_minimax_h3_image(item, caption)
    assert not is_valid_minimax_h3_text_cache(item, cache_dtype="bfloat16", require_unconditional=True)

    save_text_encoder_output_cache_minimax_h3_image(
        item, caption, unconditional_hidden_states=unconditional
    )
    assert is_valid_minimax_h3_text_cache(item, cache_dtype="bfloat16", require_unconditional=True)
    state = load_file(item.text_encoder_output_cache_path)
    assert state["varlen_mmh3_unconditional_hidden_states_bfloat16"].shape == (2, 5120)


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
