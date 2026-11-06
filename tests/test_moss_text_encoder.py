import json

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from musubi_tuner.modules.convrot_int8_utils import quantize_weight_convrot
from musubi_tuner.modules.convrot_int8_kernels import dequantize_int8_convrot_weight
from musubi_tuner.moss_image.text_encoder import _stream_from_cpu, load_text_encoder
from musubi_tuner.moss_image.text_w4a8 import decode_rotated_int8


def tiny_config(hidden=32):
    return Qwen3_5TextConfig(
        vocab_size=32, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=hidden // 4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        use_cache=False,
    )


def test_streaming_gated_attention_preloads_indirect_convolution():
    torch.manual_seed(4)
    model = Qwen3_5TextModel(tiny_config()).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).last_hidden_state
        _stream_from_cpu(model, "cpu")
        actual = model(input_ids=ids, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected)


def test_comfy_convrot_embedding_and_linear_loading(tmp_path):
    torch.manual_seed(3)
    config = tiny_config(hidden=256)
    reference = Qwen3_5TextModel(config).eval()
    state = dict(reference.state_dict())
    for path in ("embed_tokens", "layers.0.mlp.up_proj"):
        weight, scale, _ = quantize_weight_convrot(path + ".weight", state[path + ".weight"])
        state[path + ".weight"] = weight
        state[path + ".weight_scale"] = scale
        spec = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
        state[path + ".comfy_quant"] = torch.tensor(list(spec.encode()), dtype=torch.uint8)
    # Unrelated vision / LM-head quantization must not be loaded.
    state["lm_head.comfy_quant"] = torch.tensor([123, 125], dtype=torch.uint8)
    path = tmp_path / "qwen.safetensors"
    save_file({"model.language_model." + k if not k.startswith("lm_head.") else k: v
               for k, v in state.items()}, str(path))
    loaded = load_text_encoder(path, dtype=torch.float32, config=config)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = reference(input_ids=ids, use_cache=False).last_hidden_state
        actual = loaded(input_ids=ids, use_cache=False).last_hidden_state
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, expected, atol=0.12, rtol=0.12)


def test_comfy_w4a8_checkpoint_executes_using_packed_weights(tmp_path):
    torch.manual_seed(12)
    config = tiny_config(hidden=256)
    reference = Qwen3_5TextModel(config).eval()
    name = "layers.0.mlp.up_proj"
    module = reference.get_submodule(name)
    n, k = module.weight.shape
    packed = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8)
    rel = torch.full((n, k // 16), 0.5, dtype=torch.float8_e4m3fn)
    channel = torch.full((n,), 0.002, dtype=torch.float32)
    codebook = torch.arange(-8, 8, dtype=torch.float32)
    decoded = decode_rotated_int8(packed, rel, codebook, 16)
    with torch.no_grad():
        module.weight.copy_(dequantize_int8_convrot_weight(decoded, channel[:, None], 256))
    state = dict(reference.state_dict())
    state[name + ".weight"] = packed
    state[name + ".weight_s_rel"] = rel
    state[name + ".weight_s_channel"] = channel
    state[name + ".weight_codebook"] = codebook
    spec = json.dumps({"format": "asym_w4a8_int8", "group_size": 16, "convrot": True, "convrot_groupsize": 256})
    state[name + ".comfy_quant"] = torch.tensor(list(spec.encode()), dtype=torch.uint8)
    path = tmp_path / "qwen_w4a8.safetensors"
    save_file({"model.language_model." + key: value for key, value in state.items()}, str(path))
    loaded = load_text_encoder(path, dtype=torch.float32, config=config)
    assert loaded.get_submodule(name).weight.shape == packed.shape
    _stream_from_cpu(loaded, "cpu")
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = reference(input_ids=ids, use_cache=False).last_hidden_state
        actual = loaded(input_ids=ids, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
