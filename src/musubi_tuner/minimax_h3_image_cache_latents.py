"""Cache single-frame MiniMax-H3 video-VAE latents from ordinary image datasets."""

from __future__ import annotations

import argparse
import logging

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.cache_io import save_latent_cache_minimax_h3_image
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.video_vae import encode_video_target, load_video_vae
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)


def encode_and_save_batch(vae, batch: list[ItemInfo], cache_seed: int) -> None:
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    for item in batch:
        content = item.content[0] if isinstance(item.content, list) else item.content
        pixels = torch.from_numpy(content[..., :3]).permute(2, 0, 1).unsqueeze(0)
        pixels = (pixels.to(device=device, dtype=dtype) / 127.5) - 1.0
        latent = encode_video_target(vae, pixels, cache_seed, item.item_key)[0]
        logger.info("Saving MiniMax-H3 image latent %s to %s", tuple(latent.shape), item.latent_cache_path)
        save_latent_cache_minimax_h3_image(item, latent)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--cache_seed", type=int, default=42, help="deterministic VAE posterior-sampling seed")
    parser.add_argument("--disable_mmap", action="store_true", help="disable memory-mapped VAE loading")
    return parser


def main() -> None:
    args = setup_parser(cache_latents.setup_parser_common()).parse_args()
    if args.disable_cudnn_backend:
        torch.backends.cudnn.enabled = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)

    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        config_utils.load_user_config(args.dataset_config),
        args,
        architecture=ARCHITECTURE_MINIMAX_H3,
    )
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    if args.debug_mode is not None:
        cache_latents.show_datasets(
            group.datasets,
            args.debug_mode,
            args.console_width,
            args.console_back,
            args.console_num_images,
            fps=1,
        )
        return
    if not args.vae:
        raise ValueError("MiniMax-H3 video VAE checkpoint is required")
    logger.info("Loading MiniMax-H3 video VAE from %s", args.vae)
    vae = load_video_vae(args.vae, device=device, dtype=dtype, disable_mmap=args.disable_mmap)
    vae.requires_grad_(False).eval()

    def encode(batch: list[ItemInfo]):
        encode_and_save_batch(vae, batch, args.cache_seed)

    cache_latents.encode_datasets(group.datasets, encode, args)


if __name__ == "__main__":
    main()
