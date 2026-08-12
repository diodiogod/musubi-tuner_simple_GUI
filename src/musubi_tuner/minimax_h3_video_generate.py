"""Generate a short native MiniMax-H3 video with compact downstream assets."""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

from safetensors.torch import load_file
import torch
from tqdm.auto import tqdm

from musubi_tuner.minimax_h3.image_text_encoder import DEFAULT_PROCESSOR_ID, load_minimax_h3_te
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.video_sampling import decode_video_latent, sample_video_latent, write_silent_video
from musubi_tuner.minimax_h3.video_vae import load_video_vae
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.utils.device_utils import clean_memory_on_device


logger = logging.getLogger(__name__)


def _load_loras(model, paths: list[str], multipliers: list[float], device: torch.device):
    networks = []
    if len(multipliers) == 1 and len(paths) > 1:
        multipliers = multipliers * len(paths)
    if paths and len(paths) != len(multipliers):
        raise ValueError("Provide one --lora_multiplier per --network_weights, or one multiplier for all LoRAs")
    for path, multiplier in zip(paths, multipliers):
        logger.info("Loading MiniMax-H3 LoRA %s at multiplier %s", path, multiplier)
        state = load_file(path, device="cpu")
        network = lora_minimax_h3.create_arch_network_from_weights(
            float(multiplier), state, unet=model, for_inference=True
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
        raise RuntimeError("Compact MiniMax-H3 video generation currently requires CUDA")
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
        hidden_states = encoder.encode(args.prompt)[0].detach().cpu()
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

    progress = tqdm(total=args.steps, desc="MiniMax-H3 video denoising")
    try:
        latent = sample_video_latent(
            transformer,
            hidden_states,
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            device=device,
            video_shift=args.video_shift,
            audio_shift=args.audio_shift,
            step_callback=lambda _done, _total: progress.update(1),
        ).detach().cpu()
    finally:
        progress.close()
        if transformer.offloader is not None and hasattr(transformer.offloader, "copier"):
            transformer.offloader.copier.sync()

    del networks, transformer, hidden_states
    gc.collect()
    clean_memory_on_device(device)

    logger.info("Loading MiniMax-H3 VAE for native temporal decode")
    vae = load_video_vae(args.vae, device=device, dtype=torch.float16, disable_mmap=args.disable_mmap).eval()
    pixels = decode_video_latent(vae, latent.to(device=device, dtype=torch.float16), frame_count=args.frames)
    del vae, latent
    gc.collect()
    clean_memory_on_device(device)
    output = write_silent_video(pixels, args.output, fps=args.fps)
    logger.info("Saved MiniMax-H3 video preview to %s", output)
    return output


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a short native video with compact MiniMax-H3 assets")
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--text_encoder", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_PROCESSOR_ID)
    parser.add_argument("--text_encoder_load_mode", choices=("auto", "direct", "nf4"), default="auto")
    parser.add_argument("--text_encoder_blocks_to_swap", type=int, default=0)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--network_weights", action="append", default=[])
    parser.add_argument("--lora_multiplier", action="append", type=float, default=[])
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frames", type=int, default=39)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_shift", type=float, default=12.0)
    parser.add_argument("--audio_shift", type=float, default=3.0)
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
    if args.frames < 5 or (args.frames - 5) % 17:
        raise ValueError("--frames must follow MiniMax-H3's 17*n+5 frame rule (5, 22, 39, ...)")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    generate(args)


if __name__ == "__main__":
    main()
