"""Shared loaders / helpers for the Krea 2 (K2) integration."""

import logging
from typing import Optional, Union

import torch

from musubi_tuner.krea2.krea2_encoder import (
    QWEN3_VL_4B_INSTRUCT_REPO_ID,
    Qwen3VLConditioner,
    TextEncoderConfig,
    load_qwen3_vl_conditioner,
)
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, WeightTransformHooks, load_safetensors

logger = logging.getLogger(__name__)


# Dynamic fp8 quantization scope for the DiT: the per-block (SingleStreamBlock) attention
# and SwiGLU Linear weights — the heavy, repeated compute, matching the LoRA target. The
# modulation (`mod.lin`) is a raw nn.Parameter and the RMSNorm scales must stay in compute
# dtype, so both are excluded (cf. Z-Image's split). `txtfusion` (the text-fusion transformer,
# whose submodule is also named `layerwise_blocks` and so matches "blocks.") is small and
# delicate, so it is kept in compute dtype too.
KREA2_FP8_OPTIMIZATION_TARGET_KEYS = ["blocks."]
KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS = ["mod.", "norm", "txtfusion"]
FP8_SCALE_SUFFIX = ".weight_scale"
COMFY_FP8_MARKER_SUFFIX = ".comfy_quant"
KREA2_PROJECTOR_WEIGHT_KEY = "txtfusion.projector.weight"
KREA2_PROJECTOR_DIFF_KEY = "diffusion_model.txtfusion.projector.diff"


# The single config shipped with the OSS checkpoints (single_mmdit_large_wide).
single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


def _reshape_prequant_fp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Reshape a pre-quantized FP8 weight scale to broadcast against ``[out, in]``."""
    if scale.ndim == 1:
        return scale.unsqueeze(1)  # per-channel [out] -> [out, 1]
    if scale.ndim == 0:
        return scale.reshape(1)  # per-tensor [] -> [1]
    return scale


def _make_krea2_comfy_fp8_split_hook(compute_dtype: torch.dtype):
    """Normalize pre-quantized Krea 2 FP8 keys to Musubi's monkey-patch layout.

    - ``<name>.weight_scale`` -> ``<name>.scale_weight``
    - ``<name>.comfy_quant`` marker keys are dropped
    - all other keys pass through unchanged
    """

    def split_hook(key: str, value: Optional[torch.Tensor]):
        if key.endswith(COMFY_FP8_MARKER_SUFFIX):
            return [], None
        if key.endswith(FP8_SCALE_SUFFIX):
            new_key = key[: -len(FP8_SCALE_SUFFIX)] + ".scale_weight"
            if value is None:
                return [new_key], None
            return [new_key], [_reshape_prequant_fp8_scale(value).to(compute_dtype)]
        return None, None

    return split_hook


def load_krea2_projector_diff(
    path: str,
    strength: float = 1.0,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Load the tiny projector diff patch and apply the requested multiplier."""
    with MemoryEfficientSafeOpen(path) as f:
        diff = f.get_tensor(KREA2_PROJECTOR_DIFF_KEY)
    if dtype is None:
        dtype = diff.dtype
    return (diff.to(device=device, dtype=dtype) * strength).contiguous()


def apply_krea2_projector_diff_to_state_dict(
    state_dict: dict[str, torch.Tensor],
    diff_path: str,
    strength: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Apply the projector diff patch to a loaded Krea 2 state dict."""
    if diff_path is None:
        return state_dict
    if KREA2_PROJECTOR_WEIGHT_KEY not in state_dict:
        raise KeyError(f"{KREA2_PROJECTOR_WEIGHT_KEY} not found in Krea 2 state_dict")
    weight = state_dict[KREA2_PROJECTOR_WEIGHT_KEY]
    diff = load_krea2_projector_diff(diff_path, strength, device=weight.device, dtype=torch.float32)
    merged = (weight.to(torch.float32) + diff).to(weight.dtype)
    state_dict[KREA2_PROJECTOR_WEIGHT_KEY] = merged
    return state_dict


def load_krea2_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    config: SingleMMDiTConfig = single_mmdit_large_wide,
    fp8_scaled: bool = False,
    loading_device: Optional[Union[str, torch.device]] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    lora_weights: Optional[list] = None,
    lora_multipliers: Optional[list] = None,
    projector_diff_path: Optional[str] = None,
    projector_diff_strength: float = 1.0,
) -> SingleStreamDiT:
    """Build the K2 single-stream MMDiT on meta and load weights (assign=True).

    When ``fp8_scaled`` is True, the per-block Linear weights are dynamically quantized to
    scaled fp8 at load time and the matching Linear forwards are monkey-patched to
    dequantize on the fly (cf. Z-Image / qwen_image). ``dtype`` is then ignored — non-target
    weights (norms, modulation, embedders, heads) keep their checkpoint dtype.

    ``lora_weights`` (a list of loaded LoRA state dicts, with optional ``lora_multipliers``)
    are merged into the base weights at load time. This is the only correct route under fp8
    (fp8-quantized weights cannot be post-hoc merged), and it also keeps loading uniform for
    block swap: the merged/quantized state dict is produced before the model is placed, so the
    offloader can stream blocks afterward without an external weight mutation.

    For block swap, pass ``loading_device="cpu"``: the weights stay on CPU (``move_to_device``
    is then False) and the caller's ``enable_block_swap`` / ``move_to_device_except_swap_blocks``
    places the resident blocks on ``device`` and keeps the swap blocks on CPU.
    """
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)
    has_lora = lora_weights is not None and len(lora_weights) > 0
    prequant_hooks = None
    is_prequant_fp8 = False

    with MemoryEfficientSafeOpen(dit_path) as f:
        keys = f.keys()
    is_prequant_fp8 = any(k.endswith(FP8_SCALE_SUFFIX) for k in keys)
    if is_prequant_fp8:
        prequant_hooks = WeightTransformHooks(split_hook=_make_krea2_comfy_fp8_split_hook(dtype))

    logger.info(
        f"Loading Krea 2 DiT weights from {dit_path}"
        + (" (fp8 scaled)" if fp8_scaled else "")
        + (" (pre-quantized fp8)" if is_prequant_fp8 else "")
        + (f" (+{len(lora_weights)} LoRA merged)" if has_lora else "")
    )
    with torch.device("meta"):
        dit = SingleStreamDiT(config, attn_mode=attn_mode, split_attn=split_attn)

    if fp8_scaled or has_lora or is_prequant_fp8:
        # Single load path that merges LoRA (if any) into the base weights and optionally
        # quantizes the per-block Linears to scaled fp8. fp8 targets/excludes only apply when
        # quantizing; without fp8 the weights are merged and cast to ``dtype`` as-is.
        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=lora_weights,
            lora_multipliers=lora_multipliers,
            fp8_optimization=fp8_scaled or is_prequant_fp8,
            calc_device=device,
            move_to_device=(loading_device == device),
            dit_weight_dtype=None if fp8_scaled else dtype,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS if fp8_scaled else None,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS if fp8_scaled else None,
            weight_transform_hooks=prequant_hooks,
            allow_prequantized_fp8=is_prequant_fp8,
        )
        if fp8_scaled or is_prequant_fp8:
            apply_fp8_monkey_patch(dit, sd, use_scaled_mm=False)
        if projector_diff_path is not None:
            apply_krea2_projector_diff_to_state_dict(sd, projector_diff_path, projector_diff_strength)
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)
        dit.load_state_dict(sd, strict=True, assign=True)
    else:
        # Load without mmap (disable_mmap=True) to avoid the official load_file's transient ~2x
        # RAM (mmap page cache + materialized tensor), file locking, and lazy disk reads. Load
        # directly to the target device+dtype (assign=True) so the loaded tensors become the params.
        sd = load_safetensors(dit_path, device=loading_device, disable_mmap=True, dtype=dtype)
        dit.load_state_dict(sd, strict=True, assign=True)

    return dit


def load_krea2_dit_state_dict(
    dit_path: str,
    fp8_scaled: bool = False,
    calc_device: Union[str, torch.device] = "cpu",
    result_device: Union[str, torch.device] = "cpu",
    config: SingleMMDiTConfig = single_mmdit_large_wide,
    projector_diff_path: Optional[str] = None,
    projector_diff_strength: float = 1.0,
) -> dict:
    """Produce a Krea 2 DiT state dict matching a model loaded via ``load_krea2_dit``.

    Unlike ``load_krea2_dit`` this builds no ``nn.Module`` — it returns only the weights,
    for swapping the base weights of an already-built model in place (e.g. RAW-train /
    Turbo-sample). When ``fp8_scaled`` is True the per-block Linears are dynamically
    quantized exactly as in ``load_krea2_dit`` (quantization runs on ``calc_device``), so
    the returned keys include the matching ``.scale_weight`` entries and line up 1:1 with
    the live model's ``named_parameters()`` + ``named_buffers()``. The result is moved to
    ``result_device``.

    When ``result_device`` equals ``calc_device`` (e.g. both the GPU, used by the M2 turbo/raw
    swap), the dict is built straight on that device with no full intermediate CPU dict — the
    CPU peak stays at ~1 tensor. When ``result_device`` is CPU (e.g. the M1 resident stash),
    the fp8 path quantizes on ``calc_device`` and then lands the dict on CPU.
    """
    calc_dev = torch.device(calc_device)
    rd = torch.device(result_device)
    prequant_hooks = None
    is_prequant_fp8 = False
    # Keep the fp8-quantized tensors on calc_device when that is also the result device, so the
    # dict never round-trips through a full CPU copy (the M2 GPU-direct swap path).
    move_to_device = calc_dev == rd

    with MemoryEfficientSafeOpen(dit_path) as f:
        keys = f.keys()
    is_prequant_fp8 = any(k.endswith(FP8_SCALE_SUFFIX) for k in keys)
    if is_prequant_fp8:
        prequant_hooks = WeightTransformHooks(split_hook=_make_krea2_comfy_fp8_split_hook(torch.bfloat16))

    if fp8_scaled or is_prequant_fp8:
        target_keys = KREA2_FP8_OPTIMIZATION_TARGET_KEYS if fp8_scaled else None
        exclude_keys = KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS if fp8_scaled else None
        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=True,
            calc_device=calc_dev,
            move_to_device=move_to_device,
            dit_weight_dtype=None,
            target_keys=target_keys,
            exclude_keys=exclude_keys,
            weight_transform_hooks=prequant_hooks,
            allow_prequantized_fp8=is_prequant_fp8,
        )
    else:
        # Load without mmap (disable_mmap=True) to avoid the official load_file's transient ~2x
        # RAM, file locking, and lazy disk reads. Load directly to result_device in bf16.
        sd = load_safetensors(dit_path, device=result_device, disable_mmap=True, dtype=torch.bfloat16)

    if projector_diff_path is not None:
        apply_krea2_projector_diff_to_state_dict(sd, projector_diff_path, projector_diff_strength)
    sd = {k: v.to(rd) for k, v in sd.items()}
    return sd


def load_krea2_text_encoder(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
) -> Qwen3VLConditioner:
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``path`` (local safetensors,
    ComfyUI or official key layout), tokenizer from ``tokenizer_repo`` (Hub id or local dir)."""
    return load_qwen3_vl_conditioner(
        path,
        dtype=dtype,
        device=device,
        max_length=max_length,
        select_layers=select_layers,
        tokenizer_repo=tokenizer_repo,
    )


@torch.no_grad()
def get_krea2_prompt_embeds(encoder: Qwen3VLConditioner, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hiddens, mask).

    hiddens: (B, seq, num_select_layers, hidden) stacked selected hidden states.
    mask:    (B, seq) bool attention mask (valid tokens incl. suffix, padding=False).
    """
    hiddens, mask = encoder(prompts)
    return hiddens, mask.to(dtype=torch.bool)
