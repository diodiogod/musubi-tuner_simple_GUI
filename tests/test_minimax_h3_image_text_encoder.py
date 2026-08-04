import json

import pytest
import torch

from musubi_tuner.minimax_h3.image_text_encoder import _dequant_comfy_weight, _nvfp4_dequant


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
