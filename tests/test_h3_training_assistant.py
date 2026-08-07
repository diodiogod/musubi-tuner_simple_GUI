import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.training.h3_training_assistant import (
    base_preservation_loss,
    convert_ai_toolkit_weights,
    load_live_assistant,
    should_preserve_base,
)


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(4, 4, bias=False)


class DiTBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()

    def forward(self, value):
        return self.attn.qkv(value)


class _TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([DiTBlock()])

    def forward(self, value):
        return self.blocks[0](value)


def test_ai_toolkit_h3_adapter_names_convert_to_native_lora_names():
    weights = {
        "diffusion_model.blocks.0.attn.qkv.lora_A.weight": torch.zeros(16, 8),
        "diffusion_model.blocks.0.attn.qkv.lora_B.weight": torch.zeros(24, 16),
        "diffusion_model.token_refiner.blocks.1.mlp.fc1.lora_A.weight": torch.zeros(16, 8),
        "diffusion_model.token_refiner.blocks.1.mlp.fc1.lora_B.weight": torch.zeros(32, 16),
    }

    converted = convert_ai_toolkit_weights(weights)

    main = "lora_unet_blocks_0_attn_qkv"
    refiner = "lora_unet_token_refiner_blocks_1_mlp_fc1"
    assert converted[f"{main}.lora_down.weight"] is weights[next(iter(weights))]
    assert converted[f"{main}.alpha"].item() == 16
    assert converted[f"{refiner}.lora_up.weight"].shape == (32, 16)


def test_incomplete_assistant_module_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        convert_ai_toolkit_weights(
            {"diffusion_model.blocks.0.attn.qkv.lora_A.weight": torch.zeros(4, 8)}
        )


def test_sparse_base_preservation_cadence_is_one_based():
    assert not should_preserve_base(0, 10)
    assert should_preserve_base(9, 10)
    assert should_preserve_base(19, 10)
    with pytest.raises(ValueError):
        should_preserve_base(0, 0)


def test_base_preservation_reference_is_detached():
    prediction = torch.tensor([1.0, 3.0], requires_grad=True)
    reference = torch.tensor([0.0, 1.0], requires_grad=True)

    loss = base_preservation_loss(prediction, reference)
    loss.backward()

    assert prediction.grad is not None
    assert reference.grad is None


def test_live_assistant_is_frozen_toggleable_and_unmerged(tmp_path):
    path = tmp_path / "helper.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.attn.qkv.lora_A.weight": torch.ones(2, 4),
            "diffusion_model.blocks.0.attn.qkv.lora_B.weight": torch.ones(4, 2),
        },
        str(path),
    )
    transformer = _TinyTransformer()
    value = torch.ones(1, 4)
    base = transformer(value).detach().clone()

    assistant = load_live_assistant(transformer, str(path), torch.device("cpu"), torch.float32)
    active = transformer(value).detach().clone()
    assistant.set_enabled(False)
    disabled = transformer(value).detach().clone()

    assert not torch.equal(active, base)
    torch.testing.assert_close(disabled, base)
    assert all(not parameter.requires_grad for parameter in assistant.parameters())
