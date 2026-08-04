import json

import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.minimax_h3.image_text_encoder import _dequant_comfy_weight, _nvfp4_dequant
from musubi_tuner.minimax_h3.qwen3vl_text import MiniMaxQwen3VLTextConfig, MiniMaxQwen3VLTextModel
from musubi_tuner.modules.nvfp4_utils import PackedInt8Embedding, PackedNVFP4Linear


class _TensorReader:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, key):
        return self.tensors[key]


def _marker(**values):
    return torch.tensor(list(json.dumps(values).encode("utf-8")), dtype=torch.uint8)


def test_nvfp4_nibbles_and_scales_decode_to_bf16():
    # Code 2 is 1.0 and code 3 is 1.5; high nibble is the first value.
    packed = torch.full((128, 32), 0x23, dtype=torch.uint8)
    blocked_scales = torch.ones(128, 4, dtype=torch.float8_e4m3fn)
    actual = _nvfp4_dequant(packed, blocked_scales, torch.tensor(2.0))
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual[:, 0::2].float(), torch.full((128, 32), 2.0))
    torch.testing.assert_close(actual[:, 1::2].float(), torch.full((128, 32), 3.0))


def test_text_encoder_rejects_rotated_int8_weights():
    base = "model.layers.0.self_attn.q_proj"
    reader = _TensorReader(
        {
            f"{base}.comfy_quant": _marker(format="int8_tensorwise", convrot=True, convrot_groupsize=256),
        }
    )
    with pytest.raises(NotImplementedError, match="int8_convrot"):
        _dequant_comfy_weight(reader, base, {f"{base}.comfy_quant"})


def test_plain_tensorwise_int8_text_weight_dequantizes():
    base = "model.embed_tokens"
    reader = _TensorReader(
        {
            f"{base}.comfy_quant": _marker(format="int8_tensorwise", convrot=False),
            f"{base}.weight": torch.tensor([[2, -4]], dtype=torch.int8),
            f"{base}.weight_scale": torch.tensor([[0.5]], dtype=torch.float32),
        }
    )
    actual = _dequant_comfy_weight(reader, base, set(reader.tensors))
    torch.testing.assert_close(actual.float(), torch.tensor([[1.0, -2.0]]))


def test_packed_int8_embedding_dequantizes_only_selected_rows():
    layer = PackedInt8Embedding(4, 3, device="cpu")
    weight = torch.tensor([[1, 2, 3], [4, -6, 8], [9, 10, 11], [-2, 0, 2]], dtype=torch.int8)
    layer.load_quantized(weight, torch.tensor(0.5), torch.device("cpu"), torch.bfloat16)

    actual = layer(torch.tensor([[1, 3]]))

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), torch.tensor([[[2.0, -3.0, 4.0], [-1.0, 0.0, 1.0]]]))


def test_packed_int8_embedding_selects_per_row_scales():
    layer = PackedInt8Embedding(4, 2, device="cpu")
    weight = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int8)
    scales = torch.tensor([[0.5], [1.0], [1.5], [2.0]], dtype=torch.float32)
    layer.load_quantized(weight, scales, torch.device("cpu"), torch.bfloat16)

    actual = layer(torch.tensor([[2, 0, 3]]))

    expected = torch.tensor([[[7.5, 9.0], [0.5, 1.0], [14.0, 16.0]]])
    torch.testing.assert_close(actual.float(), expected)


def test_packed_nvfp4_linear_keeps_weight_packed_between_forwards():
    packed = torch.full((128, 32), 0x23, dtype=torch.uint8)
    blocked_scales = torch.ones(128, 4, dtype=torch.float8_e4m3fn)
    global_scale = torch.tensor(0.25)
    pre_quant_scale = torch.linspace(0.5, 1.5, 64, dtype=torch.bfloat16)
    layer = PackedNVFP4Linear(64, 128, device="cpu")
    layer.load_quantized(
        packed,
        blocked_scales,
        global_scale,
        torch.device("cpu"),
        torch.bfloat16,
        pre_quant_scale,
    )
    inputs = torch.randn(2, 64, dtype=torch.bfloat16)

    actual = layer(inputs)
    expected_weight = _nvfp4_dequant(packed, blocked_scales, global_scale)
    expected = F.linear(inputs * pre_quant_scale, expected_weight)

    assert layer.weight.dtype == torch.uint8
    assert tuple(layer.weight.shape) == (128, 32)
    torch.testing.assert_close(actual, expected)


def _tiny_text_model():
    config = MiniMaxQwen3VLTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=64,
        rope_dims=(1, 1, 0),
    )
    torch.manual_seed(7)
    model = MiniMaxQwen3VLTextModel(config, device="cpu", dtype=torch.float32).eval()
    for parameter in model.parameters():
        if parameter.ndim == 1:
            torch.nn.init.ones_(parameter)
        else:
            torch.nn.init.normal_(parameter, std=0.05)
    return model


def test_standalone_qwen3vl_tower_has_checkpoint_compatible_names_and_shape():
    model = _tiny_text_model()
    output = model(torch.tensor([[1, 2, 3]])).last_hidden_state

    assert output.shape == (1, 3, 16)
    names = dict(model.named_parameters())
    assert "layers.0.self_attn.q_proj.weight" in names
    assert "layers.1.mlp.down_proj.weight" in names


def test_standalone_qwen3vl_tower_is_causal():
    model = _tiny_text_model()
    first = model(torch.tensor([[1, 2, 3]])).last_hidden_state
    changed_future = model(torch.tensor([[1, 9, 10]])).last_hidden_state

    torch.testing.assert_close(first[:, 0], changed_future[:, 0], rtol=0, atol=1e-7)
