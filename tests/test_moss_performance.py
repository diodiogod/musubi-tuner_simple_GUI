"""Numerical and integration checks for the optional Fizgig-style MOSS paths."""
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.optimizers.muon import Muon
from musubi_tuner.optimizers._muon import Muon as ReferenceMuon
from musubi_tuner.optimizers.automagic4 import Automagic4
from musubi_tuner.moss_image.fast_lora import LowPrecisionLoRA
from musubi_tuner.moss_image.performance import pad_context, prepare_compile


def test_batched_muon_matches_per_tensor_updates_and_mixed_parameter_fallback():
    torch.manual_seed(20)
    ours = [torch.nn.Parameter(torch.randn(*s)) for s in [(4, 12), (4, 12), (12, 4), (3,), (2, 3, 2)]]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in ours]
    opt = Muon([{"params": ours[:2], "lr": 0.0002}, {"params": ours[2:]}], lr=0.0001)
    ref = ReferenceMuon([{"params": reference[:2], "lr": 0.0002}, {"params": reference[2:]}], lr=0.0001)
    for _ in range(4):
        for a, b in zip(ours, reference):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        opt.step()
        ref.step()
    for a, b in zip(ours, reference):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    clone = [torch.nn.Parameter(p.detach().clone()) for p in ours]
    resumed = Muon([{"params": clone[:2]}, {"params": clone[2:]}])
    import copy
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for a, b in zip(ours, clone):
        a.grad = torch.randn_like(a)
        b.grad = a.grad.clone()
    opt.step()
    resumed.step()
    for a, b in zip(ours, clone):
        torch.testing.assert_close(a, b)


def test_automagic4_checkpoint_continuation_is_consistent():
    import copy
    p = torch.nn.Parameter(torch.randn(3, 7))
    opt = Automagic4([p], lr=1e-6)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = Automagic4([q], lr=1e-6)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for _ in range(3):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        opt.step()
        resumed.step()
    torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-8)
    assert opt.get_learning_rates() == pytest.approx(resumed.get_learning_rates(), rel=1e-6)
    with pytest.raises(ValueError, match="sensitivity"):
        Automagic4([q], step_target=0.2).load_state_dict(opt.state_dict())


def test_low_precision_lora_gradients_match_explicit_bf16_compute():
    torch.manual_seed(2)
    x = torch.randn(2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
    down = torch.randn(3, 8, requires_grad=True)
    up = torch.randn(6, 3, requires_grad=True)
    reference = [v.detach().clone().requires_grad_(True) for v in (x, down, up)]
    out = LowPrecisionLoRA.apply(x, down, up, 0.75)
    rx, rd, ru = reference
    expected = torch.nn.functional.linear(torch.nn.functional.linear(rx, rd.bfloat16()), ru.bfloat16()) * 0.75
    gradient = torch.randn_like(out)
    out.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(out, expected)
    for p, r in zip((x, down, up), reference):
        torch.testing.assert_close(p.grad, r.grad)
    assert down.dtype == up.dtype == down.grad.dtype == torch.float32


def test_padding_preserves_tokens_and_attention_output():
    context = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    padded, pmask = pad_context(context, mask, 16)
    torch.testing.assert_close(padded[:, :5], context)
    torch.testing.assert_close(pmask[:, :5], mask)
    assert padded.shape[1] == 16 and pmask[:, 5:].count_nonzero() == 0
    query = torch.randn(2, 1, 3, 8)
    out = torch.nn.functional.scaled_dot_product_attention(query, context[:, None], context[:, None], mask[:, None, None].bool())
    expected = torch.nn.functional.scaled_dot_product_attention(query, padded[:, None], padded[:, None], pmask[:, None, None].bool())
    torch.testing.assert_close(out, expected)


def test_compile_refuses_swapping_without_mutating_model():
    with pytest.raises(ValueError, match="Blocks to swap"):
        prepare_compile(SimpleNamespace(compile=True), 2)


def test_compiler_primes_convrot_constants_before_tracing():
    from musubi_tuner.moss_image.performance import attach_compile_adapters
    from musubi_tuner.modules.convrot_int8_kernels import _HADAMARD_CACHE
    model = torch.nn.Sequential(torch.nn.Linear(16, 8))
    model[0]._convrot_groupsize = 4
    # No actual INT8 kernel is needed to pin down the unsafe lazy-cache side effect.
    attach_compile_adapters(model)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        matrix = _HADAMARD_CACHE[(4, "cpu", dtype)]
        assert matrix.shape == (4, 4) and matrix.dtype == dtype


def test_compiler_bridge_preserves_training_and_sampling_only_adapters(monkeypatch):
    from musubi_tuner.moss_image.performance import attach_compile_adapters
    from musubi_tuner.networks.lora import LoRAModule, LoRAInfModule
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    layer = torch.nn.Linear(8, 6).requires_grad_(False)
    training = LoRAModule("train", layer, lora_dim=2, alpha=2)
    training.apply_to()
    sampling = LoRAInfModule("preview", layer, lora_dim=2, alpha=2)
    sampling.apply_to()
    for adapter in (training, sampling):
        torch.nn.init.constant_(adapter.lora_up.weight, 0.04)
    sampling.requires_grad_(False)
    inputs = torch.randn(2, 8)
    sampling.enabled = False
    expected_off = layer(inputs).detach()
    sampling.enabled = True
    expected_on = layer(inputs).detach()
    assert not torch.equal(expected_off, expected_on)
    attach_compile_adapters(layer)
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    sampling.enabled = False
    torch.testing.assert_close(compiled(inputs), expected_off)
    sampling.enabled = True
    output = compiled(inputs)
    torch.testing.assert_close(output, expected_on)
    output.square().sum().backward()
    assert training.lora_up.weight.grad.abs().sum() > 0
    assert sampling.lora_up.weight.grad is None


@pytest.mark.parametrize("boundary,fast", [("inside", True), ("outside", True), ("inside", False)])
def test_compiled_moss_blocks_keep_adapter_gradients_and_checkpointing(boundary, fast, monkeypatch):
    # This is deliberately a CPU AOT wiring check, even on a CUDA workstation.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from musubi_tuner.moss_image.models import MossImage
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer
    from musubi_tuner.networks.lora_moss_image import create_arch_network
    torch.manual_seed(8)
    model = MossImage(dim=32, ffn_dim=48, num_heads=2, num_layers=2, text_dim=16,
                      freq_dim=16, latent_channels=128).bfloat16().requires_grad_(False)
    network = create_arch_network(1.0, 2, 2, None, [], model, moss_lowp_lora=fast)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    model.enable_gradient_checkpointing()
    model.train()
    trainer = MossImageNetworkTrainer()
    trainer.blocks_to_swap = 0
    args = SimpleNamespace(compile=True, compile_backend="aot_eager", compile_mode="default",
                           compile_fullgraph=True, compile_dynamic="false", compile_cache_size_limit=256,
                           moss_compile_checkpoint=boundary)
    trainer.compile_transformer(args, model)
    out = model(torch.randn(1, 128, 2, 2).bfloat16(),
                torch.tensor([500.0]), torch.randn(1, 5, 16).bfloat16(), None)
    out.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    assert any(p.grad.abs().sum() > 0 for p in network.parameters())
    assert all("_moss_compile_adapter" not in key for key in network.state_dict())


def test_clean_end_share_distribution_and_refresh_lr():
    from musubi_tuner.moss_image.timesteps import clean_end_policy, clean_share_mean
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, moss_setup_parser, setup_parser_common
    assert clean_end_policy("30:7:60:8:3", 7) == (30, 1.0)
    assert clean_end_policy("30:7:60:8:3", 8) == (60, 1.0)
    assert clean_end_policy("30:7:60:8:3", 11) == (30, 0.5)
    assert clean_end_policy("30:7:60:8:3", 12) == (60, 1.0)
    assert clean_share_mean(60) > clean_share_mean(40) > clean_share_mean(30)
    args = moss_setup_parser(setup_parser_common()).parse_args(
        ["--moss_pretraining_schedule", "--moss_clean_end_share", "40"])
    trainer = MossImageNetworkTrainer()
    noise = torch.ones(20000, 1, 1, 1, 1)
    _, times = trainer.get_noisy_model_input_and_timesteps(args, noise, torch.zeros_like(noise), None, None, "cpu", torch.float32)
    assert 0.38 < float((times < 500).float().mean()) < 0.42
    # Halve the actual learning rate on refresh epochs, restoring it before the scheduler.
    args.moss_clean_end_share = "30:7:60:8:3"
    trainer._moss_epoch = SimpleNamespace(value=11)
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    trainer.on_train_start(args, None, None, None, opt)
    p.grad = torch.ones_like(p)
    opt.step()
    torch.testing.assert_close(p, torch.tensor([0.95]))
    assert opt.param_groups[0]["lr"] == 0.1


def test_moss_backend_and_both_gui_schemas_expose_optional_controls():
    from backends.moss_image import build_commands, training_options
    from backends.moss_settings import DEFAULTS, PERFORMANCE_KEYS
    from modern_gui.settings import settings_schema
    settings = {**DEFAULTS, "training_mode": "MOSS-Image (Experimental)", "moss_dit_model": "base.safetensors",
                "dataset_config": "data.toml", "optimizer_type": "Muon", "learning_rate": "0.0001",
                "output_dir": ".", "output_name": "test",
                "network_type": "LoRA", "moss_fast_lora": True, "moss_text_pad_multiple": 16}
    command = build_commands(settings)[0]
    assert "moss_lowp_lora=True" in command
    assert command[command.index("--moss_text_pad_multiple") + 1] == "16"
    assert training_options(settings)["optimizer_type"] == "Muon"
    with pytest.raises(ValueError, match="LoKr"):
        training_options({**settings, "network_type": "LoKr"})
    schema = settings_schema()
    # The schema lists fields by section; flatten via its public representation.
    import json
    encoded = json.dumps(schema)
    assert all(key in encoded for key in PERFORMANCE_KEYS)
    assert '"Muon"' in encoded and '"Automagic4"' in encoded


def test_fizgig_bare_muon_args_reach_the_optimizer_without_changing_defaults():
    from backends.moss_image import build_commands
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, moss_setup_parser, setup_parser_common
    settings = {"output_dir": ".", "output_name": "x", "optimizer_type": "Muon", "learning_rate": "0.0001",
                "optimizer_args": "adjust_lr_fn=match_rms_adamw weight_decay=0.01 momentum=0.95 nesterov=True ns_steps=5"}
    command = build_commands(settings)[0]
    script = command.index("src/musubi_tuner/moss_image_train_network.py")
    args = moss_setup_parser(setup_parser_common()).parse_args(command[script+1:])
    _, _, opt, _, _ = MossImageNetworkTrainer().get_optimizer(args, [torch.nn.Parameter(torch.ones(2, 3))])
    assert opt.defaults["adjust_lr_fn"] == "match_rms_adamw"
    assert opt.defaults["weight_decay"] == 0.01
