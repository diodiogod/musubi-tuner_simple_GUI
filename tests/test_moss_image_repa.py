from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Dinov2Config, Dinov2Model

from musubi_tuner.moss_image.models import MossImage
from musubi_tuner.moss_image.repa import MossRepa, validate_config
from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, moss_setup_parser, setup_parser_common
from musubi_tuner.networks.lora_moss_image import create_arch_network


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0

    def decode(self, latent):
        self.calls += 1
        return torch.nn.functional.interpolate(latent[:, :3] * self.weight, size=(28, 28))


def make_repa():
    args = moss_setup_parser(setup_parser_common()).parse_args([
        "--moss_repa_enabled", "--moss_repa_block", "0", "--moss_repa_image_size", "28",
        "--moss_repa_cache_size", "1",
    ])
    encoder = Dinov2Model(Dinov2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                      intermediate_size=32, image_size=28, patch_size=14))
    return args, MossRepa(args, 32, "cpu", encoder=encoder)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_alignment_backpropagates_to_lora_and_head_but_not_teacher_or_vae(checkpointing):
    torch.manual_seed(10)
    args, repa = make_repa()
    model = MossImage(dim=32, ffn_dim=48, num_heads=2, num_layers=2,
                      text_dim=24, freq_dim=16, latent_channels=128).requires_grad_(False)
    network = create_arch_network(1, 2, 2, None, [], model)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    if checkpointing:
        model.enable_gradient_checkpointing()
    model.train()
    vae = TinyVAE()
    latents = torch.randn(1, 128, 1, 2, 3)  # rectangular training bucket
    target = repa.targets(vae, latents)
    assert target.shape == (1, 4, 16)
    features = {}
    inputs = torch.randn(1, 128, 2, 3, requires_grad=checkpointing)
    model(inputs, torch.tensor([500.]), torch.randn(1, 3, 24),
          feature_layer=0, feature_buffer=features)
    assert features["image"].shape == (1, 6, 32)  # excludes three caption tokens
    before = repa.projector[1].weight.detach().clone()
    loss, logs = repa.loss(features.pop("image"), target, 2, 3)
    loss.backward()  # REPA alone must reach adapters; diffusion loss cannot hide a missing path
    assert loss.isfinite() and "repa/cosine_similarity" in logs
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())
    assert repa.projector[1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in repa.encoder.parameters())
    assert vae.weight.grad is None
    assert all(p.grad is None for p in model.parameters())
    torch.optim.AdamW(repa.projector.parameters(), lr=0.01).step()
    assert not torch.equal(before, repa.projector[1].weight)


def test_cpu_cache_reuses_exact_latents_and_is_bounded():
    _, repa = make_repa()
    vae = TinyVAE()
    latents = torch.randn(1, 128, 1, 2, 3)
    first = repa.targets(vae, latents)
    torch.testing.assert_close(first, repa.targets(vae, latents.clone()))
    assert vae.calls == 1
    repa.targets(vae, latents + 1)
    assert vae.calls == 2 and len(repa.cache) == 1
    repa.targets(vae, latents)
    assert vae.calls == 3  # first entry was evicted
    repa.args.moss_repa_cache_size = 0
    repa.cache.clear()
    repa.targets(vae, latents)
    repa.targets(vae, latents)
    assert not repa.cache and vae.calls == 5


def test_projection_head_round_trip_and_incompatible_resume_rejected():
    _, first = make_repa()
    _, second = make_repa()
    second.load_state_dict(first.state_dict())
    for a, b in zip(first.projector.parameters(), second.projector.parameters()):
        torch.testing.assert_close(a, b)
    second.config["moss_repa_block"] = 1
    with pytest.raises(ValueError, match="fresh optimizer"):
        second.load_state_dict(first.state_dict())


@pytest.mark.parametrize("key,value", [("moss_repa_weight", float("nan")), ("moss_repa_weight", 0),
                                       ("moss_repa_block", 42), ("moss_repa_image_size", 225),
                                       ("moss_repa_cache_size", -1), ("moss_repa_encoder", "")])
def test_invalid_configuration_rejected_before_loading_weights(key, value):
    args, _ = make_repa()
    setattr(args, key, value)
    with pytest.raises(ValueError):
        validate_config(args)


def test_dop_forward_does_not_capture_repa_features():
    args, repa = make_repa()
    trainer = MossImageNetworkTrainer()
    trainer._repa = repa
    calls = []

    def transformer(*inputs, **kwargs):
        calls.append(kwargs)
        return torch.zeros(1, 128, 2, 2)

    args.gradient_checkpointing = False
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    latent = torch.zeros(1, 128, 1, 2, 2)
    trainer.call_dit(args, accelerator, transformer, latent, {"moss_embed": [torch.ones(1, 4096)]},
                     latent, latent, torch.tensor([500.]), torch.float32)
    assert calls == [{}]


def test_gui_settings_reach_parser_and_detailed_help():
    from backends.moss_image import build_commands
    from backends.moss_settings import DEFAULTS, MODE
    from modern_gui.settings import settings_schema
    from modern_gui.validation import validate_training_settings
    settings = dict(DEFAULTS, training_mode=MODE, moss_repa_enabled=True, moss_repa_device="cpu",
                    output_dir="out", output_name="repa")
    command = build_commands(settings)[0]
    args = moss_setup_parser(setup_parser_common()).parse_args(command[command.index("src/musubi_tuner/moss_image_train_network.py") + 1:])
    assert args.moss_repa_enabled and args.moss_repa_device == "cpu" and args.moss_repa_weight == .5
    fields = {field["key"]: field for section in settings_schema(settings)["sections"] for field in section["fields"]}
    assert "fresh optimizer" in fields["moss_repa_enabled"]["help_detail"]
    assert fields["moss_repa_enabled"]["modes"] == [MODE]
    settings["moss_repa_image_size"] = "225"
    assert any("REPA" in item["message"] for item in validate_training_settings(settings)["errors"])
    settings["moss_repa_enabled"] = False
    assert not any("--moss_repa" in str(arg) for arg in build_commands(settings)[0])
