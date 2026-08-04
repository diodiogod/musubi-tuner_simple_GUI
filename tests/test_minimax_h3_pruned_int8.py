import json

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.checkpoint import inspect_convrot_int8_checkpoint, load_safetensors_module
from musubi_tuner.minimax_h3.model import AdalnProj, MiniMaxH3Config, MiniMaxH3Model
from musubi_tuner.networks.lora_minimax_h3 import create_arch_network
from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import patch_convrot_int8_modules


def _marker(**overrides) -> torch.Tensor:
    payload = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    payload.update(overrides)
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


class _TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(256, 32, bias=False, dtype=torch.bfloat16)


def test_direct_prequantized_loader_preserves_int8_and_backward(tmp_path):
    torch.manual_seed(1)
    source = _TinyLinear()
    quantized, scale = quantize_int8_convrot_weight(source.proj.weight.detach().float(), 256)
    checkpoint = tmp_path / "tiny_convrot.safetensors"
    save_file(
        {
            "proj.weight": quantized,
            "proj.weight_scale": scale,
            "proj.comfy_quant": _marker(),
        },
        checkpoint,
    )

    scale_shapes, manifest = inspect_convrot_int8_checkpoint([checkpoint])
    assert scale_shapes == {"proj": torch.Size([32, 1])}
    assert manifest["proj.weight"][1] == torch.int8

    def factory():
        model = _TinyLinear()
        return patch_convrot_int8_modules(model, scale_shapes)

    model = load_safetensors_module(
        factory,
        [checkpoint],
        device="cpu",
        dtype=None,
        key_transform=lambda key: "proj.scale_weight" if key == "proj.weight_scale" else key,
        ignore_key=lambda key: key.endswith(".comfy_quant"),
        strict_dtype=True,
        allow_quantized=True,
    )
    assert model.proj.weight.dtype == torch.int8
    assert model.proj.weight.requires_grad is False
    assert model.proj.scale_weight.dtype == torch.float32

    inputs = torch.randn(2, 256, dtype=torch.bfloat16, requires_grad=True)
    model.proj(inputs).float().square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_convrot_contract_rejects_wrong_groupsize(tmp_path):
    checkpoint = tmp_path / "bad_convrot.safetensors"
    save_file(
        {
            "proj.weight": torch.zeros(32, 256, dtype=torch.int8),
            "proj.weight_scale": torch.ones(32, 1, dtype=torch.float32),
            "proj.comfy_quant": _marker(convrot_groupsize=128),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="Unsupported MiniMax-H3 quantization metadata"):
        inspect_convrot_int8_checkpoint([checkpoint])


def test_pruned_adaln_uses_curve_basis_without_silu():
    projection = AdalnProj(2, 1, expand=1, modalities=1, apply_silu=False, dtype=torch.float16)
    with torch.no_grad():
        projection.linear.weight.copy_(torch.tensor([[2.0, -1.0]], dtype=torch.float16))
        projection.linear.bias.zero_()
    timestep_basis = torch.tensor([[0.5, -0.25]], dtype=torch.bfloat16)
    (actual,) = projection(timestep_basis)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), torch.tensor([[1.25]]), atol=1e-3, rtol=0)


def test_pruned_model_has_table_instead_of_full_time_embedder():
    config = MiniMaxH3Config(
        hidden_size=256,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=512,
        text_dim=48,
        time_embed_hidden_size=32,
        time_embed_dim=8,
        rope_inv_freq_len=2,
        adaln_curve_grid=17,
    )
    model = MiniMaxH3Model(config, dtype=torch.bfloat16, device="meta")
    assert "adaln_t_table" in model.state_dict()
    assert not any(key.startswith("time_embedder.") for key in model.state_dict())
    assert model.blocks[0].adaln_proj.linear.weight.shape == (256 * 6 * 3, 8)
    assert model.blocks[0].adaln_proj.linear.weight.dtype == torch.float16


def test_image_only_lora_backward_through_pruned_convrot_base():
    config = MiniMaxH3Config(
        hidden_size=256,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=512,
        text_dim=48,
        time_embed_hidden_size=32,
        time_embed_dim=8,
        rope_inv_freq_len=2,
        adaln_curve_grid=17,
    )
    model = MiniMaxH3Model(config, dtype=torch.bfloat16, device="cpu")
    target_paths = {
        f"blocks.0.{suffix}"
        for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    }
    quantized = {}
    for name, module in model.named_modules():
        if name in target_paths:
            quantized[name] = quantize_int8_convrot_weight(module.weight.detach().float(), 256)
    patch_convrot_int8_modules(
        model,
        {name: scale.shape for name, (_, scale) in quantized.items()},
    )
    with torch.no_grad():
        for name, (weight, scale) in quantized.items():
            module = model.get_submodule(name)
            module.weight.copy_(weight)
            module.scale_weight.copy_(scale)
        model.rope.inv_freq.copy_(torch.tensor([1.0, 0.01]))
        model.adaln_t_table.normal_()
    model.requires_grad_(False)

    network = create_arch_network(1.0, 4, 4.0, None, [], model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    assert len(network.unet_loras) == 4

    latent = torch.randn(1, 24, 1, 2, 2, dtype=torch.bfloat16)
    text = torch.randn(1, 2, 48, dtype=torch.bfloat16)
    prediction = model.forward_image(latent, torch.tensor([0.4]), text)
    assert prediction.shape == latent.shape
    prediction.float().square().mean().backward()
    grads = [parameter.grad for parameter in network.parameters() if parameter.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)
