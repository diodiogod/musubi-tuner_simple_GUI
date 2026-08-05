"""Cache compact Qwen3-VL-32B layer-50 states for H3 image-only training."""

from __future__ import annotations

import argparse
import logging

import torch
from safetensors import safe_open

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_minimax_h3_image
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.image_text_encoder import DEFAULT_PROCESSOR_ID, load_minimax_h3_te
from musubi_tuner.training.dop import (
    add_cache_arguments,
    dop_signature,
    is_valid_dop_cache,
    make_class_caption,
    validate_dop_config,
)


logger = logging.getLogger(__name__)


def is_valid_minimax_h3_text_cache(
    item: ItemInfo,
    dop_trigger_word: str = "",
    dop_class_word: str = "",
    cache_dtype: str = "float32",
) -> bool:
    """Accept caption-matching caches produced by the corrected Comfy-style tower."""
    path = str(getattr(item, "text_encoder_output_cache_path", "") or "")
    if not path:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as cache:
            metadata = cache.metadata() or {}
            keys = list(cache.keys())
            if metadata.get("caption1", "") != str(getattr(item, "caption", "") or ""):
                return False
            if f"varlen_mmh3_hidden_states_{cache_dtype}" not in keys:
                return False
    except (OSError, RuntimeError, ValueError):
        return False
    if dop_trigger_word or dop_class_word:
        return is_valid_dop_cache(
            item,
            dop_trigger_word,
            dop_class_word,
            f"varlen_dop_mmh3_hidden_states_{cache_dtype}",
        )
    return True


def encode_and_save_batch(
    encoder,
    batch: list[ItemInfo],
    dop_trigger_word: str = "",
    dop_class_word: str = "",
    cache_dtype: torch.dtype = torch.bfloat16,
) -> None:
    use_dop = bool(dop_trigger_word or dop_class_word)
    signature = None
    if use_dop:
        validate_dop_config(dop_trigger_word, dop_class_word)
        signature = dop_signature(dop_trigger_word, dop_class_word)
    for item in batch:
        logger.info("Encoding MiniMax-H3 caption for %s", item.item_key)
        hidden_states = encoder.encode(item.caption)[0].to(dtype=cache_dtype)
        dop_hidden_states = None
        if use_dop:
            try:
                class_caption = make_class_caption(item.caption, dop_trigger_word, dop_class_word)
            except ValueError as exc:
                raise ValueError(f"DOP caption error for {item.item_key}: {exc}") from exc
            logger.info("Encoding MiniMax-H3 DOP class caption for %s", item.item_key)
            dop_hidden_states = encoder.encode(class_caption)[0].to(dtype=cache_dtype)
        save_text_encoder_output_cache_minimax_h3_image(item, hidden_states, dop_hidden_states, signature)


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
    parser.add_argument(
        "--cache_dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help=(
            "dtype used to store the final caption embeddings; bfloat16 is recommended for smaller caches and "
            "lower training-time I/O, while the text encoder itself still computes in float32"
        ),
    )
    add_cache_arguments(parser)
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
        compute_dtype=torch.float32,
        quantize=True,
        tokenizer_dir=args.tokenizer,
        load_mode=args.text_encoder_load_mode,
    )

    cache_dtype = torch.bfloat16 if args.cache_dtype == "bfloat16" else torch.float32

    def encode(batch: list[ItemInfo]):
        encode_and_save_batch(encoder, batch, args.dop_trigger_word, args.dop_class_word, cache_dtype)

    # Precision is part of the cache contract so changing the UI option rebuilds only
    # caption caches while keeping image latents untouched.
    cache_validator = lambda item: is_valid_minimax_h3_text_cache(
        item, args.dop_trigger_word, args.dop_class_word, args.cache_dtype
    )

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files,
        all_cache_paths,
        encode,
        cache_validator=cache_validator,
    )
    del encoder
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()
