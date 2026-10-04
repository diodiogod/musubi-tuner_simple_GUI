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
from musubi_tuner.minimax_h3_native.text_encoder import (
    H3Presentation,
    IMAGE_PLACEHOLDER,
    encode_h3_presentation,
    load_h3_processor,
    load_h3_text_encoder,
    wrap_ref_teacher_caption,
    wrap_subject_reference_caption,
)
from musubi_tuner.training.dop import (
    add_cache_arguments,
    dop_signature,
    is_valid_dop_cache,
    make_class_caption,
    validate_dop_config,
)


logger = logging.getLogger(__name__)


class _NativeTextAdapter:
    """Expose the compact encoder's tiny ``encode`` contract over the full visual tower."""

    def __init__(self, processor, encoder) -> None:
        self.processor = processor
        self.encoder = encoder

    def encode(self, text: str):
        presentation = H3Presentation(text=text, processor_text=text)
        hidden_states, _ = encode_h3_presentation(self.processor, self.encoder, presentation)
        return (hidden_states,)


def is_valid_minimax_h3_text_cache(
    item: ItemInfo,
    dop_trigger_word: str = "",
    dop_class_word: str = "",
    cache_dtype: str = "float32",
    require_unconditional: bool = False,
    require_teacher: bool = False,
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
            if require_unconditional and f"varlen_mmh3_unconditional_hidden_states_{cache_dtype}" not in keys:
                return False
            if require_teacher:
                if f"varlen_mmh3_teacher_ref_hidden_states_{cache_dtype}" not in keys:
                    return False
                if "varlen_mmh3_teacher_ref_token_tags_int64" not in keys:
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
    unconditional_hidden_states: torch.Tensor | None = None,
    teacher_encoder=None,
    teacher_processor=None,
    teacher_conditions: str | None = None,
) -> None:
    use_dop = bool(dop_trigger_word or dop_class_word)
    signature = None
    if use_dop:
        validate_dop_config(dop_trigger_word, dop_class_word)
        signature = dop_signature(dop_trigger_word, dop_class_word)
    for item in batch:
        logger.info("Encoding MiniMax-H3 caption for %s", item.item_key)
        hidden_states = encoder.encode(item.caption)[0].to(dtype=cache_dtype)
        teacher_hidden_states = teacher_token_tags = None
        if teacher_encoder is not None:
            if teacher_conditions == "subject_ref":
                if not item.control_content:
                    raise ValueError(
                        "Other-subject reference learning requires control_path/control_path_N images in an image JSONL dataset"
                    )
                images = tuple(torch.as_tensor(reference)[..., :3] for reference in item.control_content)
                wrapped = wrap_subject_reference_caption(item.caption, len(images))
            else:
                frames = torch.as_tensor(item.content)
                if frames.ndim == 3:
                    frames = frames.unsqueeze(0)
                if frames.ndim != 4 or frames.shape[0] != 1:
                    raise ValueError(
                        f"Compact H3 reference-guided caching requires one decoded image [1,H,W,C], got {tuple(frames.shape)}"
                    )
                images = (frames[0],)
                wrapped = wrap_ref_teacher_caption(item.caption)
            prefix = "".join(
                f"<Picture {index}>: {IMAGE_PLACEHOLDER}" for index in range(1, len(images) + 1)
            )
            presentation_text = prefix + wrapped
            teacher_hidden_states, teacher_token_tags = encode_h3_presentation(
                teacher_processor,
                teacher_encoder,
                H3Presentation(
                    text=presentation_text,
                    images=images,
                    processor_text=presentation_text,
                ),
            )
            teacher_hidden_states = teacher_hidden_states.to(dtype=cache_dtype)
        dop_hidden_states = None
        if use_dop:
            try:
                class_caption = make_class_caption(item.caption, dop_trigger_word, dop_class_word)
            except ValueError as exc:
                raise ValueError(f"DOP caption error for {item.item_key}: {exc}") from exc
            logger.info("Encoding MiniMax-H3 DOP class caption for %s", item.item_key)
            dop_hidden_states = encoder.encode(class_caption)[0].to(dtype=cache_dtype)
        save_text_encoder_output_cache_minimax_h3_image(
            item,
            hidden_states,
            dop_hidden_states,
            signature,
            unconditional_hidden_states,
            teacher_hidden_states,
            teacher_token_tags,
        )


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
        "--text_encoder_blocks_to_swap",
        type=int,
        default=0,
        help="Stream 0-50 Qwen language layers from system RAM; 50 uses the least VRAM",
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
    parser.add_argument(
        "--cache_h3_unconditional",
        action="store_true",
        help="Also cache the empty-prompt state required by H3 guidance-distillation protection",
    )
    parser.add_argument(
        "--teacher_conditions",
        choices=("ref", "subject_ref"),
        default=None,
        help=(
            "Cache a visual teacher presentation for experimental compact reference-guided learning. "
            "'ref' lets the frozen teacher see the same training image; ordinary caption-only caching stays unchanged."
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

    teacher_encoder = teacher_processor = None
    if args.teacher_conditions:
        logger.info("Loading visual-capable MiniMax-H3 Qwen3-VL teacher encoder from %s", args.text_encoder)
        teacher_processor = load_h3_processor()
        teacher_encoder = load_h3_text_encoder(
            args.text_encoder,
            device=device,
            dtype=torch.bfloat16,
            blocks_to_swap=args.text_encoder_blocks_to_swap,
        )
        encoder = None
    else:
        logger.info("Loading MiniMax-H3 Qwen3-VL-32B text encoder from %s", args.text_encoder)
        encoder = load_minimax_h3_te(
            args.text_encoder,
            device=device,
            compute_dtype=torch.float32,
            quantize=True,
            tokenizer_dir=args.tokenizer,
            load_mode=args.text_encoder_load_mode,
            blocks_to_swap=args.text_encoder_blocks_to_swap,
        )

    cache_dtype = torch.bfloat16 if args.cache_dtype == "bfloat16" else torch.float32
    unconditional_hidden_states = None
    if args.cache_h3_unconditional:
        logger.info("Encoding MiniMax-H3 empty prompt for guidance-distillation protection")
        if encoder is None:
            empty = H3Presentation(text="", processor_text="")
            unconditional_hidden_states = encode_h3_presentation(teacher_processor, teacher_encoder, empty)[0].to(dtype=cache_dtype)
        else:
            unconditional_hidden_states = encoder.encode("")[0].to(dtype=cache_dtype)

    def encode(batch: list[ItemInfo]):
        nonlocal encoder, teacher_encoder
        encode_and_save_batch(
            encoder if encoder is not None else _NativeTextAdapter(teacher_processor, teacher_encoder),
            batch,
            args.dop_trigger_word,
            args.dop_class_word,
            cache_dtype,
            unconditional_hidden_states,
            teacher_encoder,
            teacher_processor,
            args.teacher_conditions,
        )

    # Precision is part of the cache contract so changing the UI option rebuilds only
    # caption caches while keeping image latents untouched.
    cache_validator = lambda item: is_valid_minimax_h3_text_cache(
        item, args.dop_trigger_word, args.dop_class_word, args.cache_dtype, args.cache_h3_unconditional,
        bool(args.teacher_conditions),
    )

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files,
        all_cache_paths,
        encode,
        requires_content=bool(args.teacher_conditions),
        cache_validator=cache_validator,
        cache_args=args,
    )
    del encoder, teacher_encoder
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()
