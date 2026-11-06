"""Cache MOSS Qwen3.5 captions and optional DOP class captions."""
import logging
import torch

from musubi_tuner import cache_text_encoder_outputs as caching
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ImageDataset
from musubi_tuner.moss_image.cache import save_embeddings
from musubi_tuner.moss_image.text_encoder import load_text_encoder, load_tokenizer, encode_prompts
from musubi_tuner.training.dop import (
    add_cache_arguments, dop_signature, is_valid_dop_cache, make_class_caption, validate_dop_config,
)


def setup_parser():
    parser = caching.setup_parser_common()
    parser.add_argument("--text_encoder", required=True)
    parser.add_argument("--tokenizer_cache_dir")
    parser.add_argument("--moss_max_token_length", type=int, default=4096)
    parser.add_argument("--text_encoder_cpu_streaming", action="store_true")
    parser.add_argument("--disable_mmap", action="store_true")
    add_cache_arguments(parser)
    return parser


def encode_and_save_batch(encoder, tokenizer, batch, args):
    use_dop = bool(args.dop_trigger_word or args.dop_class_word)
    signature = dop_signature(args.dop_trigger_word, args.dop_class_word) if use_dop else None
    # Save each caption immediately; a large dataset batch never accumulates
    # hundreds of full-length embeddings in GPU memory.
    for item in batch:
        embed = encode_prompts(encoder, tokenizer, [item.caption], args.moss_max_token_length)[0]
        class_embed = None
        if use_dop:
            caption = make_class_caption(item.caption, args.dop_trigger_word, args.dop_class_word)
            class_embed = encode_prompts(encoder, tokenizer, [caption], args.moss_max_token_length)[0]
        save_embeddings(item, embed, class_embed, signature)


def main():
    args = setup_parser().parse_args()
    if args.dop_trigger_word or args.dop_class_word:
        validate_dop_config(args.dop_trigger_word, args.dop_class_word)
    if not 1 <= args.moss_max_token_length <= 4096:
        raise ValueError("MOSS maximum token length must be between 1 and 4096.")
    config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(config, args, architecture=ARCHITECTURE_MOSS_IMAGE)
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
    if any(not isinstance(dataset, ImageDataset) for dataset in datasets):
        raise ValueError("MOSS-Image requires image datasets.")
    files, paths = caching.prepare_cache_files_and_paths(datasets)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = load_tokenizer(args.tokenizer_cache_dir)
    encoder = load_text_encoder(args.text_encoder, device=device,
                                cpu_streaming=args.text_encoder_cpu_streaming, disable_mmap=args.disable_mmap)
    validator = None
    if args.dop_trigger_word or args.dop_class_word:
        validator = lambda item: is_valid_dop_cache(
            item, args.dop_trigger_word, args.dop_class_word, "varlen_dop_moss_embed_",
        )
    caching.process_text_encoder_batches(
        args.num_workers, args.skip_existing, args.batch_size, datasets, files, paths,
        lambda batch: encode_and_save_batch(encoder, tokenizer, batch, args),
        cache_validator=validator, cache_args=args,
    )
    del encoder
    caching.post_process_cache_files(datasets, files, paths, args.keep_cache)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
