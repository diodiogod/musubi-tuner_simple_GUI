"""Standalone compact MiniMax-H3 still-image generation for LoRA testing."""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm.auto import tqdm

from musubi_tuner.minimax_h3.image_sampling import decode_image_latent, sample_image_latent
from musubi_tuner.minimax_h3.image_text_encoder import DEFAULT_PROCESSOR_ID, load_minimax_h3_te
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.video_vae import load_video_vae
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.utils.device_utils import clean_memory_on_device


logger = logging.getLogger(__name__)


def _load_loras(model, paths: list[str], multipliers: list[float], device: torch.device):
    networks = []
    if not paths:
        return networks
    if len(multipliers) == 1 and len(paths) > 1:
        multipliers = multipliers * len(paths)
    if len(paths) != len(multipliers):
        raise ValueError("Provide one --lora_multiplier per --network_weights, or one multiplier for all LoRAs")
    for path, multiplier in zip(paths, multipliers):
        logger.info("Loading MiniMax-H3 LoRA %s at multiplier %s", path, multiplier)
        state = load_file(path, device="cpu")
        network = lora_minimax_h3.create_arch_network_from_weights(
            float(multiplier),
            state,
            unet=model,
            for_inference=True,
        )
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        info = network.load_state_dict(state, strict=True)
        if info.missing_keys or info.unexpected_keys:
            raise ValueError(
                f"MiniMax-H3 LoRA load mismatch for {path}: missing={info.missing_keys[:8]}, "
                f"unexpected={info.unexpected_keys[:8]}"
            )
        network.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        networks.append(network)
    return networks


def generate(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Compact MiniMax-H3 image generation currently requires CUDA")
    device = torch.device(args.device or "cuda")

    logger.info("Encoding MiniMax-H3 prompt before allocating the DiT")
    encoder = load_minimax_h3_te(
        args.text_encoder,
        device=device,
        compute_dtype=torch.float32,
        quantize=True,
        tokenizer_dir=args.tokenizer,
        load_mode=args.text_encoder_load_mode,
        blocks_to_swap=args.text_encoder_blocks_to_swap,
    )
    with torch.no_grad():
        hidden_states = encoder.encode(args.prompt)[0].detach().to("cpu")
    del encoder
    gc.collect()
    clean_memory_on_device(device)

    loading_device = "cpu" if args.blocks_to_swap > 0 else device
    logger.info("Loading pruned ConvRot INT8 MiniMax-H3 transformer from %s", args.dit)
    transformer = load_h3_transformer(
        args.dit,
        device=loading_device,
        dtype=torch.bfloat16,
        attn_mode=args.attn_mode,
        split_attn=args.split_attn,
        disable_mmap=args.disable_mmap,
        convrot_bwd_mode="bf16",
    ).eval().requires_grad_(False)
    if not transformer._has_convrot_int8:
        raise ValueError("This generator requires the pruned ConvRot INT8 MiniMax-H3 transformer")

    networks = _load_loras(transformer, args.network_weights, args.lora_multiplier, device)
    if args.blocks_to_swap > 0:
        transformer.enable_block_swap(
            args.blocks_to_swap,
            BlockSwapConfig(
                device=device,
                supports_backward=False,
                use_pinned_memory=args.use_pinned_memory_for_block_swap,
                h2d_only=True,
                ring_size=args.block_swap_ring_size,
            ),
        )
        transformer.move_to_device_except_swap_blocks(device)
        transformer.switch_block_swap_for_inference()
    else:
        transformer.to(device)

    progress = tqdm(total=args.steps, desc="MiniMax-H3 denoising")
    try:
        latent = sample_image_latent(
            transformer,
            hidden_states,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            shift=args.shift,
            device=device,
            dtype=torch.bfloat16,
            step_callback=lambda _done, _total: progress.update(1),
        ).detach().to("cpu")
    finally:
        progress.close()
        if transformer.offloader is not None and hasattr(transformer.offloader, "copier"):
            transformer.offloader.copier.sync()

    del networks, transformer, hidden_states
    gc.collect()
    clean_memory_on_device(device)

    logger.info("Loading MiniMax-H3 VAE for image decode")
    vae = load_video_vae(
        args.vae,
        device=device,
        dtype=torch.float16,
        disable_mmap=args.disable_mmap,
    ).eval().requires_grad_(False)
    with torch.no_grad():
        pixels = decode_image_latent(
            vae,
            latent.to(device=device, dtype=torch.float16),
            single_frame=args.image_vae_mode == "single_frame",
        )[0, :, 0].cpu()
    del vae, latent
    clean_memory_on_device(device)

    image = Image.fromarray((pixels.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).round().astype("uint8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    logger.info("Saved MiniMax-H3 image to %s", output)
    return output


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a still image with compact MiniMax-H3 and optional LoRA")
    parser.add_argument("--dit", required=True, help="pruned ConvRot INT8 FL2VA transformer")
    parser.add_argument("--vae", required=True, help="MiniMax-H3 video VAE")
    parser.add_argument(
        "--image_vae_mode",
        choices=("temporal_compat", "single_frame"),
        default="temporal_compat",
        help="Decode mode: temporal_compat duplicates T=1 for the official video VAE; single_frame passes T=1 directly for an image-specialized H3 VAE.",
    )
    parser.add_argument("--text_encoder", required=True, help="compact Qwen3-VL NVFP4/AWQ text encoder")
    parser.add_argument("--tokenizer", default=DEFAULT_PROCESSOR_ID)
    parser.add_argument("--text_encoder_load_mode", choices=("auto", "direct", "nf4"), default="auto")
    parser.add_argument("--text_encoder_blocks_to_swap", type=int, default=0)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--network_weights", action="append", default=[])
    parser.add_argument("--lora_multiplier", action="append", type=float, default=[])
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift", type=float, default=12.0)
    parser.add_argument("--blocks_to_swap", type=int, default=30)
    parser.add_argument("--block_swap_ring_size", type=int, default=2)
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    parser.add_argument("--attn_mode", choices=("torch", "flash", "sageattn", "xformers", "flash3"), default="torch")
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--disable_mmap", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = setup_parser().parse_args()
    if not args.lora_multiplier:
        args.lora_multiplier = [1.0]
    if not 0 <= args.blocks_to_swap <= 48:
        raise ValueError("--blocks_to_swap must be between 0 and 48")
    generate(args)


if __name__ == "__main__":
    main()
