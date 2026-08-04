# Copyright 2026 Fizgig contributors.
# Licensed under the Apache License, Version 2.0.
# Adapted for Musubi from shootthesound/Fizgig v3.2.0 (src/fizgig/minimax/embedder.py).

"""MiniMax H3 text encoder: Qwen3-VL-32B language model (truncated to 50 layers).

The DiT is conditioned on the **unnormalized hidden state after language layer 50** — the
last-layer output of the truncated checkpoint, with NO final norm (comfy applies none). We
build a transformers Qwen3Model (its layout matches the checkpoint's `model.layers.*` 1:1),
strip the `model.` prefix, skip the vision tower (`visual.*` — unused for text-only captions),
NF4-quantize the big linears so the ~32B fits a 24-32 GB card, and replace the final norm with
Identity so last_hidden_state IS the raw layer-50 output.

Image LoRA training uses text-only captions (no vision blocks), tokenized raw with NO special
tokens — the H3 convention. The tokenizer is loaded from Qwen3-VL-32B or a user-provided
local processor directory.

Output: [1, L, 5120] bf16, exactly what the DiT's condition_proj expects.
"""

import torch
import torch.nn as nn

# The official safetensors safe_open(device="cpu") + get_tensor path memory-maps the file and
# slices the torch storage per tensor — on Windows, reading a 48 GB file that way hard-crashes
# (access violation, exit 0xC0000005) deep in torch.storage.__getitem__. The repo's own
# MemoryEfficientSafeOpen reads each tensor with a plain np.fromfile (no torch mmap-view), which
# is exactly why every large-model loader (krea2 / klein) uses it. Use it here too.
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

# Derived from the checkpoint tensor shapes (U8 weights are 4-bit-packed: real in-dim = 2x).
DEFAULT_PROCESSOR_ID = "Qwen/Qwen3-VL-32B-Instruct"

_QWEN3_32B_TRUNC50 = dict(
    hidden_size=5120, num_hidden_layers=50, num_attention_heads=64, num_key_value_heads=8,
    head_dim=128, intermediate_size=25600, vocab_size=151936, max_position_embeddings=262144,
    rms_norm_eps=1e-6, rope_theta=5000000.0, attention_bias=False, tie_word_embeddings=False,
)

# The Qwen3 decoder Linears to NF4 (the matmul bulk). embed_tokens stays as-is (int8 in the
# checkpoint / bf16 after load); the tiny q/k norms + layernorms stay bf16.
_NF4_SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                 ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
                 ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")


# ---------------------------------------------------------------------------
# ComfyUI "comfy_quant" dequant (nvfp4 + int8_tensorwise) — lets the small nvfp4-awq TE
# (~15 GB) stand in for the 48 GB bf16 file. We dequantize each weight back to bf16 per tensor
# at load, then NF4-quantize it on GPU exactly like the bf16 path, so nothing downstream changes.
#
# nvfp4 (comfy/float.py): W = fp4_e2m1 * block_scale * global_scale, block size 16 along the input
# dim. Stored as: weight U8 [out, in/2] (2 E2M1 values/byte, HIGH nibble = first/even element),
# weight_scale FP8-E4M3 [out, in/16] in ComfyUI's cuBLAS `to_blocked` swizzle (128x4 tiles ->
# (-1,32,16); see comfy/float.py::to_blocked — must be inverted to row-major before use),
# weight_scale_2 F32 scalar. AWQ layers (o_proj/down_proj) also carry pre_quant_scale [in]: comfy
# applies it as `input = input * pre_quant_scale` (ops.py), i.e. y=(x*s)@W^T = x@(W*s)^T — so we
# fold it into the weight columns by MULTIPLYING. For the other Linears (q/k/v, gate/up) the AWQ
# scale was folded into the PRECEDING norm weight at quantization time, so the checkpoint is
# self-consistent: load its norm weights as-is and no unfold is needed (validated per-column
# against the bf16 file: ratio W_dq/W_ref == ln_ref/ln_nvfp4 to <1%, all layer shapes).
# int8_tensorwise: W = int8 * weight_scale (per-tensor scalar).
# ---------------------------------------------------------------------------
# E2M1 magnitude for the low 3 bits (sign is bit 3): {0, .5, 1, 1.5, 2, 3, 4, 6}.
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _from_blocked(blocked, rows, cols):
    """Invert ComfyUI's to_blocked (comfy/float.py): (-1, 32, 16) tiles -> row-major [rows, cols].
    Verified as an exact roundtrip of to_blocked for the shapes in this checkpoint."""
    nrb = -(-rows // 128)
    ncb = -(-cols // 4)
    x = blocked.reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(nrb, ncb, 128, 4).permute(0, 2, 1, 3).reshape(nrb * 128, ncb * 4)
    return x[:rows, :cols].contiguous()


def _nvfp4_dequant(packed, block_scale_fp8, global_scale):
    """packed U8 [out, in/2] -> bf16 [out, in].  W = e2m1(code) * block_scale * global_scale."""
    out, in2 = packed.shape
    inp = in2 * 2
    lo = (packed & 0x0F).to(torch.long)
    hi = (packed >> 4).to(torch.long)
    codes = torch.stack([hi, lo], dim=-1).reshape(out, inp)       # HIGH nibble first (verified)
    sign = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    vals = sign * _E2M1_MAG.to(codes.device)[codes & 0x7]         # decoded fp4, f32
    bs = _from_blocked(block_scale_fp8.to(torch.float32).reshape(-1, 32, 16), out, inp // 16)
    bs = bs.repeat_interleave(16, dim=1)                          # [out, in/16] -> [out, in]
    return (vals * bs * global_scale.to(torch.float32)).to(torch.bfloat16)


def _dequant_comfy_weight(f, file_mod, ckpt):
    """Dequantize one comfy-quant Linear weight (nvfp4 or int8_tensorwise) back to bf16 [out, in].

    The `.comfy_quant` blob names the scheme — checked explicitly, because e.g. the
    int8_convrot TE variant stores ROTATED weights that would sail through a naive
    weight*scale dequant and silently produce a garbage encoder."""
    import json as _json
    fmt = ""
    try:
        blob = bytes(f.get_tensor(file_mod + ".comfy_quant").tolist())
        marker = _json.loads(blob.decode("utf-8"))
        fmt = marker.get("format", "")
        if marker.get("convrot"):
            fmt = "int8_convrot"
    except Exception:
        pass
    if fmt not in ("nvfp4", "int8_tensorwise"):
        raise NotImplementedError(
            f"Unsupported comfy-quant format '{fmt or 'unknown'}' on {file_mod} — this MiniMax-H3 cache path can load "
            "the nvfp4-awq or bf16 Qwen3-VL-32B TE. The int8_convrot variant stores rotated "
            "weights and is not supported; use qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors "
            "(~15.7 GB) or the bf16 file instead.")
    if fmt == "nvfp4":
        packed = f.get_tensor(file_mod + ".weight")
        bscale = f.get_tensor(file_mod + ".weight_scale")
        gscale = f.get_tensor(file_mod + ".weight_scale_2")
        w = _nvfp4_dequant(packed, bscale, gscale)
        pqs_key = file_mod + ".pre_quant_scale"
        if pqs_key in ckpt:                               # AWQ: fold s into the weight columns
            pqs = f.get_tensor(pqs_key).to(torch.float32)
            w = (w.to(torch.float32) * pqs.unsqueeze(0)).to(torch.bfloat16)
        return w
    # int8_tensorwise: W = int8 * weight_scale (per-tensor scalar)
    w = f.get_tensor(file_mod + ".weight").to(torch.float32)
    s = f.get_tensor(file_mod + ".weight_scale").to(torch.float32)
    return (w * s).to(torch.bfloat16)


def build_qwen3_te(config_overrides=None):
    """A Qwen3Model text encoder with no final norm (returns the raw layer-50 output)."""
    from transformers import Qwen3Config, Qwen3Model
    cfg = dict(_QWEN3_32B_TRUNC50)
    if config_overrides:
        cfg.update(config_overrides)
    model = Qwen3Model(Qwen3Config(**cfg))
    model.norm = nn.Identity()          # comfy applies NO final norm to the layer-50 conditioning
    return model


class MiniMaxH3TextEncoder:
    """Loads the bf16 TE, NF4 on GPU, and encodes captions to [1, L, 5120] bf16."""

    def __init__(self, model, tokenizer, device="cuda", compute_dtype=torch.bfloat16):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.compute_dtype = compute_dtype

    @torch.no_grad()
    def encode(self, caption: str, max_length: int = 512) -> torch.Tensor:
        # H3: raw prompt text, NO special tokens (no chat template).
        ids = self.tokenizer(caption, add_special_tokens=False, return_tensors="pt",
                             truncation=True, max_length=max_length)["input_ids"].to(self.device)
        if ids.shape[1] == 0:                              # empty caption -> single pad token
            ids = torch.tensor([[self.tokenizer.pad_token_id or 151643]], device=self.device)
        out = self.model(input_ids=ids)                    # norm=Identity -> raw layer-50 output
        return out.last_hidden_state.to(self.compute_dtype)   # [1, L, 5120]


def load_minimax_h3_te(path: str, device="cuda", compute_dtype=torch.bfloat16,
                       quantize=True, tokenizer_dir=None) -> MiniMaxH3TextEncoder:
    """Build + NF4-load the Qwen3-VL-32B language TE from the bf16 checkpoint (visual.* skipped)."""
    from bitsandbytes.nn import Linear4bit, Params4bit
    # Import Qwen before entering the meta-device context. Importing transformers
    # itself constructs a few scalar tensors and PyTorch 2.6 cannot call item()
    # on those if the global default device is already meta.
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model  # noqa: F401

    with torch.device("meta"):
        model = build_qwen3_te()

        # Swap the NF4-target Linears for Linear4bit shells INSIDE the meta context. Outside it,
        # each Linear4bit eagerly allocates a full fp32 CPU weight (nn.Linear default) — across the
        # 32B Qwen3-VL that is well over 100 GB of throwaway tensors the allocator then holds for
        # the whole caching pass. On meta the shells are 0 bytes; real weights stream in below.
        if quantize:
            for mod_name, module in list(model.named_modules()):
                for child_name, child in list(module.named_children()):
                    full = f"{mod_name}.{child_name}" if mod_name else child_name
                    if isinstance(child, nn.Linear) and (full + ".weight").endswith(_NF4_SUFFIXES):
                        q = Linear4bit(child.in_features, child.out_features, bias=child.bias is not None,
                                       compute_dtype=compute_dtype, quant_type="nf4")
                        setattr(module, child_name, q)

    dev = torch.device(device)
    if quantize and dev.type != "cuda":
        raise ValueError("MiniMax-H3 Qwen3-VL NF4 text caching requires a CUDA device")
    model_keys = {n for n, _ in model.named_parameters()}
    loaded_keys = set()
    with MemoryEfficientSafeOpen(path) as f:
        ckpt = set(f.keys())
        # The comfy nvfp4-awq TE (15 GB) stores packed quant weights + scales instead of plain
        # bf16 — detect once, then dequantize each Linear weight from its
        # `<mod>.weight/.weight_scale[/_2]/.pre_quant_scale` family back to bf16. Everything
        # else (norms, embed) loads as usual; the checkpoint's norm weights already carry the
        # folded AWQ scales, so the model function matches the bf16 TE up to 4-bit noise.
        is_comfy_quant = any(k.endswith(".comfy_quant") for k in ckpt)
        if is_comfy_quant:
            print("[minimax-te] comfy-quant checkpoint (nvfp4-awq) — dequantizing to bf16 per tensor")
        for name in model_keys:
            src = "model." + name                          # checkpoint prefixes the LM with model.
            file_mod = src.rsplit(".", 1)[0]
            leaf = name.rsplit(".", 1)[1]
            if is_comfy_quant and leaf == "weight" and (file_mod + ".comfy_quant") in ckpt:
                w = _dequant_comfy_weight(f, file_mod, ckpt)   # -> bf16 [out, in]
            elif src in ckpt:
                w = f.get_tensor(src)
            else:
                continue                                   # e.g. norm (Identity) has no params
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            if quantize and (name.endswith(_NF4_SUFFIXES)):
                p = Params4bit(w.to(compute_dtype), requires_grad=False, quant_type="nf4").to(dev)
                setattr(parent, leaf, p)
            else:
                keep = w.to(torch.float32) if w.dtype == torch.float32 else w.to(compute_dtype)
                setattr(parent, leaf, nn.Parameter(keep.to(dev), requires_grad=False))
            loaded_keys.add(name)

    missing = sorted(model_keys - loaded_keys)
    if missing:
        raise ValueError(f"MiniMax-H3 text encoder checkpoint is missing model tensors: {missing[:20]}")

    # Computed (non-checkpoint) buffers stayed on meta from the meta-build. The rotary
    # embedding's inv_freq is the load-bearing one — rebuild it on the real device. A general
    # sweep materializes any other stray meta buffers to be safe.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    model.rotary_emb = Qwen3RotaryEmbedding(model.config).to(dev)
    for mod in model.modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf is not None and buf.is_meta:
                mod.register_buffer(bname, torch.zeros(buf.shape, dtype=buf.dtype, device=dev))
    model.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(tokenizer_dir or DEFAULT_PROCESSOR_ID)
    return MiniMaxH3TextEncoder(model, tok, device=device, compute_dtype=compute_dtype)
