"""Cache compact Qwen3-VL-32B layer-50 states for H3 image-only training."""

from __future__ import annotations

import argparse
import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_minimax_h3_image
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.image_text_encoder import DEFAULT_PROCESSOR_ID, load_minimax_h3_te


logger = logging.getLogger(__name__)


def encode_and_save_batch(encoder, batch: list[ItemInfo]) -> None:
    for item in batch:
        logger.info("Encoding MiniMax-H3 caption for %s", item.item_key)
        hidden_states = encoder.encode(item.caption)[0]
        save_text_encoder_output_cache_minimax_h3_image(item, hidden_states)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder",
        required=True,
        help="Qwen3-VL-32B safetensors; the compact Comfy nvfp4-awq file is supported and recommended",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_PROCESSOR_ID,
        help="Qwen3-VL-32B tokenizer repo or local directory",
    )
    parser.add_argument(
        "--text_encoder_load_mode",
        choices=("auto", "direct", "nf4"),
        default="auto",
        help=(
            "auto keeps compact Comfy NVFP4/INT8 weights packed for fast startup when Triton is available; "
            "nf4 selects the slower legacy conversion path"
        ),
    )
    return parser


def main() -> None:
    args = setup_parser(cache_text_encoder_outputs.setup_parser_common()).parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise ValueError("MiniMax-H3 compact text caching currently requires CUDA and bitsandbytes NF4")

    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        config_utils.load_user_config(args.dataset_config),
        args,
        architecture=ARCHITECTURE_MINIMAX_H3,
    )
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = group.datasets
    all_cache_files, all_cache_paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    logger.info("Loading MiniMax-H3 Qwen3-VL-32B text encoder from %s", args.text_encoder)
    encoder = load_minimax_h3_te(
        args.text_encoder,
        device=device,
        compute_dtype=torch.bfloat16,
        quantize=True,
        tokenizer_dir=args.tokenizer,
        load_mode=args.text_encoder_load_mode,
    )

    def encode(batch: list[ItemInfo]):
        encode_and_save_batch(encoder, batch)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files,
        all_cache_paths,
        encode,
    )
    del encoder
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()
