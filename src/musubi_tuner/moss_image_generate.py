"""Standalone MOSS-Image preview with optional Musubi-format LoRA."""
import argparse
import gc
import logging
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.moss_image.checkpoint import load_model
from musubi_tuner.moss_image.sampling import generate_latents, parse_skip_layers
from musubi_tuner.moss_image.sampling_adapter import load_sampling_adapter, normalize_lora_keys, sampling_adapter_enabled
from musubi_tuner.moss_image.text_encoder import load_text_encoder, load_tokenizer, encode_prompts, pad_text_embeds
from musubi_tuner.moss_image.vae import load_vae
from musubi_tuner.networks import lora_moss_image
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)


def preview_attention_mode(requested):
    from musubi_tuner.modules import attention
    available = {"torch": True, "flash": attention.flash_attn_varlen_func is not None,
                 "sageattn": attention.sageattn_varlen is not None, "xformers": attention.xops is not None}
    if not available.get(requested, False):
        logger.warning("Preview attention %s is unavailable; using PyTorch SDPA.", requested)
        return "torch"
    return requested


def generate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("MOSS previews currently require an NVIDIA CUDA GPU.")
    index = (max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
             if args.gpu_id == "auto" else int(args.gpu_id))
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"GPU index {index} is unavailable.")
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    logger.info("MOSS preview using GPU %d: %s", index, torch.cuda.get_device_name(index))
    logger.info("MOSS preview recipe: %d steps, %s / %s, CFG %s, sampling-only adapter %s at %s",
                args.steps, args.sampler, args.scheduler, args.cfg_scale, args.sample_lora or "off", args.sample_lora_multiplier)
    if not 0 <= args.blocks_to_swap <= 40:
        raise ValueError("MOSS previews support between 0 and 40 swapped blocks.")
    args.attn_mode = preview_attention_mode(args.attn_mode)
    encoder = load_text_encoder(args.text_encoder, device=device, cpu_streaming=args.text_encoder_cpu_streaming)
    tokenizer = load_tokenizer(args.tokenizer_cache_dir)
    embeddings = encode_prompts(encoder, tokenizer, [args.prompt, args.negative_prompt], args.moss_max_token_length)
    positive, mask = pad_text_embeds([embeddings[0]])
    negative, negative_mask = pad_text_embeds([embeddings[1]])
    del encoder, tokenizer, embeddings
    gc.collect()
    clean_memory_on_device(device)
    model = load_model(args.dit, device="cpu" if args.blocks_to_swap else device,
                       convrot_int8=args.convrot_int8, convrot_bwd_mode=args.convrot_bwd_mode,
                       attn_mode=args.attn_mode, split_attn=args.split_attn)
    networks = []
    if args.network_weights:
        state = normalize_lora_keys(load_file(args.network_weights, device="cpu"))
        if any(".lokr_w1" in key for key in state):
            from musubi_tuner.networks import lokr as network_module
        elif any(".hada_w1_a" in key for key in state):
            from musubi_tuner.networks import loha as network_module
        else:
            network_module = lora_moss_image
        network = network_module.create_arch_network_from_weights(args.lora_multiplier, state, unet=model, for_inference=True)
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        network.load_state_dict(state, strict=True)
        network.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        networks.append(network)
    extra = load_sampling_adapter(args.sample_lora, args.sample_lora_multiplier, model, device) if args.sample_lora else None
    if args.blocks_to_swap:
        model.enable_block_swap(args.blocks_to_swap, BlockSwapConfig(
            device=device, supports_backward=False, use_pinned_memory=False, h2d_only=True,
        ))
        model.move_to_device_except_swap_blocks(device)
        model.switch_block_swap_for_inference()
    with sampling_adapter_enabled(extra):
        latents = generate_latents(model, positive, mask, negative, negative_mask, args.height, args.width,
                                   args.steps, args.cfg_scale, args.flow_shift, parse_skip_layers(args.skip_layers),
                                   args.seed, device, sampler=args.sampler, scheduler=args.scheduler).cpu()
    if model.offloader is not None and hasattr(model.offloader, "copier"):
        model.offloader.copier.sync()
    del model, networks, extra
    gc.collect()
    clean_memory_on_device(device)
    vae = load_vae(args.vae, dtype=torch.float32, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        pixels = vae.decode(latents.to(device)).float().clamp(-1, 1).add(1).mul(127.5)
    array = pixels[0].permute(1, 2, 0).round().byte().cpu().numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output)
    logger.info("Saved MOSS preview to %s", output)
    return output


def setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dit", "vae", "text_encoder", "prompt", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--tokenizer_cache_dir")
    parser.add_argument("--text_encoder_cpu_streaming", action="store_true")
    parser.add_argument("--moss_max_token_length", type=int, default=4096)
    parser.add_argument("--network_weights")
    parser.add_argument("--lora_multiplier", type=float, default=1.0)
    parser.add_argument("--sample_lora")
    parser.add_argument("--sample_lora_multiplier", type=float, default=0.6)
    parser.add_argument("--sampler", choices=("euler", "ddim"), default="euler")
    parser.add_argument("--scheduler", choices=("official", "simple"), default="official")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument("--skip_layers", default="22")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--convrot_int8", action="store_true")
    parser.add_argument("--convrot_bwd_mode", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--attn_mode", choices=("torch", "flash", "sageattn", "xformers"), default="torch")
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--gpu_id", default="auto")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate(setup_parser().parse_args())
