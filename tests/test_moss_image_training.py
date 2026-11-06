from contextlib import nullcontext
from types import SimpleNamespace

import torch
import pytest

from musubi_tuner.moss_image.models import MossImage
from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, moss_setup_parser, setup_parser_common
from musubi_tuner.networks.lora_moss_image import create_arch_network, create_arch_network_from_weights


def test_training_backward_updates_only_adapter_gradients():
    model = MossImage(dim=32, ffn_dim=48, num_heads=2, num_layers=2,
                      text_dim=24, freq_dim=16, latent_channels=128).requires_grad_(False)
    network = create_arch_network(1.0, 2, 2, None, [], model)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    trainer = MossImageNetworkTrainer()
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    latents = torch.randn(1, 128, 1, 2, 2)
    noise = torch.randn_like(latents)
    batch = {"moss_embed": [torch.randn(3, 24)]}
    output = trainer.call_dit(SimpleNamespace(gradient_checkpointing=False, moss_pretraining_schedule=False, timestep_sampling="sigma"), accelerator,
                              model, latents, batch, noise, (latents + noise) / 2,
                              torch.tensor([500.0]), torch.float32)
    assert output.pred.shape == latents.shape
    torch.testing.assert_close(output.target, noise - latents)
    (output.pred - output.target).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())
    assert all(p.grad is None for p in model.parameters())


def test_parser_retains_shared_resume_and_dop_options():
    parser = moss_setup_parser(setup_parser_common())
    options = parser._option_string_actions
    assert all(key in options for key in ("--resume", "--dop_loss_weight", "--text_encoder", "--convrot_bwd_mode"))


def test_depth_loads_vae_even_without_sample_prompts():
    trainer = MossImageNetworkTrainer()
    vae = torch.nn.Linear(2, 2)
    trainer.load_vae = lambda args, dtype, path: vae
    args = SimpleNamespace(sample_prompts=None, depth_anchor_weight=0.1, vae="flux2-vae.safetensors")
    prompts, loaded = trainer._prepare_sampling(args, SimpleNamespace(), torch.float32)
    assert prompts is None and loaded is vae
    assert not vae.training and all(not p.requires_grad for p in vae.parameters())


def test_dop_repositions_swapped_blocks_after_the_no_grad_teacher():
    from musubi_tuner.training.dop import dop_signature

    class SwapModel:
        blocks_to_swap = 2
        ready = True
        resets = 0

        def prepare_block_swap_before_forward(self):
            self.ready = True
            self.resets += 1

    class Network:
        multiplier = 1.0

        def set_multiplier(self, value):
            self.multiplier = value

    trainer = MossImageNetworkTrainer()
    model, network = SwapModel(), Network()
    accelerator = SimpleNamespace(unwrap_model=lambda item: item)
    args = SimpleNamespace(dop_loss_weight=0.1, dop_trigger_word="subject", dop_class_word="a person")
    batch = {"moss_embed": torch.ones(1, 1), "dop_moss_embed": torch.ones(1, 1),
             "dop_signature": dop_signature("subject", "a person").unsqueeze(0)}
    latent = torch.ones(1, 1)
    trainer._dop_step_context = (batch, latent, latent, latent, torch.ones(1), torch.float32)

    def fake_forward(_args, _accelerator, transformer, *unused):
        assert transformer.ready
        if not torch.is_grad_enabled():
            transformer.ready = False  # no backward will restore the layout
        return SimpleNamespace(pred=torch.ones(1, 1) * (1 + network.multiplier))

    trainer.call_dit = fake_forward
    loss, _ = trainer.compute_auxiliary_loss(args, accelerator, model, network)
    assert loss.item() == pytest.approx(0.1)
    assert model.resets == 1


def test_moss_pretraining_timestep_matches_actual_noise_mix():
    trainer = MossImageNetworkTrainer()
    args = SimpleNamespace(moss_pretraining_schedule=True, moss_logit_mean=-1.09, moss_logit_std=.95,
                           moss_timestep_cutoff=.95, moss_timestep_uniform_ratio=0.0)
    count = 4096
    clean = torch.zeros(count, 1, 1, 1, 1)
    noise = torch.ones_like(clean)
    torch.manual_seed(8)
    noisy, times = trainer.get_noisy_model_input_and_timesteps(args, noise, clean, None, None, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(noisy.flatten(), times / 1000)
    assert 0.60 < (times / 1000).mean() < 0.85
    assert times.min() >= 1 and times.max() <= 1000
    torch.testing.assert_close(times, times.round(), atol=0.0001, rtol=0)


def test_continuous_sampler_one_based_time_is_corrected_for_moss():
    args = SimpleNamespace(moss_pretraining_schedule=False, timestep_sampling="uniform")
    times = torch.tensor([1.0, 501.0, 1001.0])
    torch.testing.assert_close(MossImageNetworkTrainer._model_timesteps(args, times), torch.tensor([0., 500., 1000.]))
    args.timestep_sampling = "sigma"
    torch.testing.assert_close(MossImageNetworkTrainer._model_timesteps(args, times), times)


def test_adapter_export_reload_retains_effect_and_separate_comfy_qkv(tmp_path):
    config = dict(dim=32, ffn_dim=48, num_heads=2, num_layers=2, text_dim=24, freq_dim=16, latent_channels=8)
    torch.manual_seed(13)
    first = MossImage(**config).eval().requires_grad_(False)
    second = MossImage(**config).eval().requires_grad_(False)
    second.load_state_dict(first.state_dict())
    net = create_arch_network(1.0, 2, 2, None, [], first)
    net.apply_to([], first, apply_text_encoder=False, apply_unet=True)
    for name, parameter in net.named_parameters():
        if "lora_up" in name:
            torch.nn.init.normal_(parameter, std=.02)
    path = tmp_path / "lora.safetensors"
    net.save_weights(str(path), torch.float32, {})
    from safetensors.torch import load_file
    weights = load_file(str(path))
    assert all(any(f"lora_unet_layers_0_proj_{p}.lora_down.weight" == key for key in weights) for p in ("q", "k", "v"))
    restored = create_arch_network_from_weights(1.0, weights, unet=second, for_inference=True)
    restored.apply_to([], second, apply_text_encoder=False, apply_unet=True)
    restored.load_state_dict(weights, strict=True)
    x, text, times = torch.randn(1, 8, 2, 2), torch.randn(1, 3, 24), torch.tensor([500.0])
    with torch.no_grad():
        torch.testing.assert_close(first(x, times, text), second(x, times, text))


def test_attention_only_preset_excludes_feed_forward_projections():
    model = MossImage(dim=32, ffn_dim=48, num_heads=2, num_layers=2,
                      text_dim=24, freq_dim=16, latent_channels=8)
    network = create_arch_network(1.0, 2, 2, None, [], model,
                                  exclude_patterns="['.*[.]ffn_up_gate','.*[.]ffn_down']")
    assert len(network.unet_loras) == 8
    assert all(not any(name in item.lora_name for name in ("ffn_up_gate", "ffn_down"))
               for item in network.unet_loras)


@pytest.mark.parametrize("network_type", ["lokr", "loha"])
def test_other_adapter_types_detect_moss_block_scope(network_type):
    import importlib
    module = importlib.import_module("musubi_tuner.networks." + network_type)
    model = MossImage(dim=32, ffn_dim=48, num_heads=2, num_layers=2,
                      text_dim=24, freq_dim=16, latent_channels=8).requires_grad_(False)
    network = module.create_arch_network(1.0, 2, 2, None, [], model)
    assert len(network.unet_loras) == 12
    assert all(item.lora_name.startswith("lora_unet_layers_") for item in network.unet_loras)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    x, context = torch.randn(1, 8, 2, 2), torch.randn(1, 3, 24)
    model(x, torch.tensor([500.0]), context).square().mean().backward()
    assert any(parameter.grad is not None for parameter in network.parameters())
