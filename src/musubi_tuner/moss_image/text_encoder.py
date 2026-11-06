"""Frozen Qwen3.5 language tower and official MOSS prompt formatting.

Unlike a conventional Transformer, Qwen3.5 contains recurrent attention blocks.
Encode captions separately: padding can change their hidden states.
"""
import json
import logging
from pathlib import Path

import torch
from accelerate import cpu_offload, init_empty_weights

from musubi_tuner.utils.hf_cache import from_pretrained_cache_first
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, WeightTransformHooks, get_split_weight_filenames
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
from musubi_tuner.modules.convrot_int8_kernels import dequantize_int8_convrot_weight
from musubi_tuner.moss_image.text_w4a8 import patch_w4a8_linear

MAX_TOKEN_LENGTH = 4096
logger = logging.getLogger(__name__)


def format_prompt(prompt):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def load_tokenizer(cache_dir=None):
    from transformers import AutoTokenizer

    if cache_dir and (Path(cache_dir) / "tokenizer_config.json").is_file():
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True)
    else:
        tokenizer = from_pretrained_cache_first(
            AutoTokenizer, "OpenMOSS-Team/MOSS-Image-Base", subfolder="tokenizer", cache_dir=cache_dir,
        )
    if len(tokenizer) < 1000 or not tokenizer.encode(format_prompt("test")):
        raise ValueError("The MOSS tokenizer is incomplete. Select a complete tokenizer directory or download it again.")
    return tokenizer


def normalize_key(key):
    for prefix in ("model.language_model.", "language_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _text_hook(expected, key, value, blocked_modules=frozenset()):
    """Expose only language tower tensors, including Comfy quantization companions."""
    name = normalize_key(key)
    if name.rsplit(".", 1)[0] in blocked_modules:
        return [], [] if value is not None else None
    weight = name
    for suffix in (".comfy_quant", ".weight_scale"):
        if name.endswith(suffix):
            weight = name[:-len(suffix)] + ".weight"
            break
    if weight not in expected:
        return [], [] if value is not None else None
    if name != key:
        return [name], [value] if value is not None else None
    return None, None


def _patch_quantized_embedding(model, state, groupsizes, dtype):
    key = "embed_tokens.scale_weight"
    if key not in state:
        return
    embedding = model.embed_tokens
    groupsize = groupsizes["embed_tokens"]
    embedding.register_buffer("scale_weight", torch.empty_like(state[key], device="meta"))

    def forward(self, ids):
        # Dequantize just the token rows actually used. Materializing the entire
        # 248k-token BF16 table would defeat low-VRAM text caching.
        rows = torch.nn.functional.embedding(ids, self.weight)
        scales = torch.nn.functional.embedding(ids, self.scale_weight)
        original_shape = rows.shape
        result = dequantize_int8_convrot_weight(rows.reshape(-1, original_shape[-1]), scales.reshape(-1, 1), groupsize)
        return result.reshape(original_shape).to(dtype=dtype)

    embedding.forward = forward.__get__(embedding, type(embedding))


def _stream_from_cpu(model, device):
    # GatedDeltaNet reads conv1d.weight directly instead of calling conv1d().
    # Preloading this class is required or Accelerate leaves that weight on meta.
    cpu_offload(model, execution_device=torch.device(device), offload_buffers=True,
                preload_module_classes=["Qwen3_5GatedDeltaNet"])


def load_text_encoder(path, *, device="cpu", dtype=torch.bfloat16, cpu_streaming=False, disable_mmap=False, config=None):
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    config = config or Qwen3_5TextConfig(**json.loads(Path(__file__).with_name("qwen35_config.json").read_text()))
    config.use_cache = False
    with init_empty_weights():
        # Keep non-persistent rotary buffers at their original FP32 precision.
        model = Qwen3_5TextModel(config)
    expected = set(model.state_dict())
    loaded = set()
    target = "cpu" if cpu_streaming else device
    files = get_split_weight_filenames(str(path)) or [str(path)]
    quantized = False
    grouped = {}
    for filename in files:
        with MemoryEfficientSafeOpen(filename, disable_numpy_memmap=disable_mmap) as reader:
            for key in reader.keys():
                if not key.endswith(".comfy_quant") or _text_hook(expected, key, None)[0] == []:
                    continue
                spec = json.loads(bytes(reader.get_tensor(key).tolist()).decode("utf-8"))
                name = normalize_key(key[:-len(".comfy_quant")])
                if spec.get("format") == "asym_w4a8_int8":
                    if name in grouped:
                        raise ValueError(f"Duplicate W4A8 text encoder layer: {name}")
                    grouped[name] = (filename, key[:-len(".comfy_quant")], spec)
                elif spec.get("format") != "int8_tensorwise" or not spec.get("convrot"):
                    raise ValueError(f"Unsupported Qwen3.5 text encoder quantization: {spec.get('format')}")
                quantized = True
    if quantized:
        from functools import partial
        quantizer = ConvRotInt8Quantizer(target_layer_keys=[])
        try:
            state = quantizer.load_and_quantize(
                files, calc_device="cpu", disable_numpy_memmap=disable_mmap,
                weight_transform_hooks=WeightTransformHooks(split_hook=partial(_text_hook, expected, blocked_modules=set(grouped))),
            )
        except ValueError as exc:
            raise ValueError("Unable to load the quantized Qwen3.5 text encoder. " + str(exc)) from exc
        for name, (filename, original_name, spec) in grouped.items():
            with MemoryEfficientSafeOpen(filename, disable_numpy_memmap=disable_mmap) as reader:
                keys = set(reader.keys())
                for source, destination in (("weight", "weight"), ("weight_s_rel", "w4_scale"),
                                            ("weight_s_channel", "w4_channel"), ("weight_codebook", "w4_codebook")):
                    original = original_name + "." + source
                    if original not in keys:
                        if source == "weight_codebook":
                            continue
                        raise ValueError(f"Missing W4A8 tensor: {original}")
                    value = reader.get_tensor(original)
                    if source == "weight_s_rel" and value.dtype == torch.uint8:
                        value = value.view(torch.float8_e4m3fn)
                    state[name + "." + destination] = value
            patch_w4a8_linear(model.get_submodule(name), state, name, spec)
        _patch_quantized_embedding(model, state, quantizer.module_groupsizes, dtype)
        # The common monkey patch handles Linear modules; embedding is patched above.
        linear_state = {k: v for k, v in state.items() if k != "embed_tokens.scale_weight"}
        apply_convrot_int8_monkey_patch(model, linear_state, groupsize_map=quantizer.module_groupsizes)
        for key, value in state.items():
            if key.endswith(".w4_scale"):
                state[key] = value.to(device=target)
            elif value.is_floating_point():
                target_dtype = torch.float32 if key.endswith((".scale_weight", ".w4_channel", ".w4_codebook")) else dtype
                state[key] = value.to(device=target, dtype=target_dtype)
            else:
                state[key] = value.to(target)
        loaded = set(state) & expected
    else:
        state = {}
        for filename in files:
            with MemoryEfficientSafeOpen(filename, disable_numpy_memmap=disable_mmap) as reader:
                for original in reader.keys():
                    key = normalize_key(original)
                    if key not in expected:
                        continue
                    if key in loaded:
                        raise ValueError(f"Duplicate text encoder tensor: {key}")
                    value = reader.get_tensor(original)
                    if not value.is_floating_point() or value.element_size() == 1:
                        raise ValueError(f"Unsupported text encoder quantization for {original}")
                    state[key] = value.to(device=target, dtype=dtype)
                    loaded.add(key)
    missing = sorted(expected - loaded)
    if missing:
        raise ValueError(f"Incomplete Qwen3.5 text encoder; missing {missing[:8]}")
    model.requires_grad_(False)
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    model.moss_execution_device = torch.device(device)
    if cpu_streaming and torch.device(device).type != "cpu":
        _stream_from_cpu(model, device)
    else:
        model.to(device)
    return model


@torch.no_grad()
def encode_prompts(model, tokenizer, prompts, max_length=MAX_TOKEN_LENGTH):
    if not 1 <= max_length <= MAX_TOKEN_LENGTH:
        raise ValueError("MOSS maximum token length must be between 1 and 4096.")
    device = getattr(model, "moss_execution_device", next(model.parameters()).device)
    outputs = []
    for prompt in prompts:
        ids = tokenizer(format_prompt(prompt), return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]
        output = model(input_ids=ids.to(device), use_cache=False).last_hidden_state
        outputs.append(output[0].detach().cpu())
    return outputs


def pad_text_embeds(embeddings):
    if not embeddings or any(e.ndim != 2 or e.shape[0] == 0 for e in embeddings):
        raise ValueError("Expected nonempty MOSS text embeddings with shape [tokens, channels].")
    length = max(e.shape[0] for e in embeddings)
    output = embeddings[0].new_zeros(len(embeddings), length, embeddings[0].shape[-1])
    mask = torch.zeros(len(embeddings), length, device=output.device, dtype=torch.int32)
    for index, embed in enumerate(embeddings):
        output[index, :len(embed)] = embed
        mask[index, :len(embed)] = 1
    return output, mask
