"""Cache packed, normalized Flux.2 VAE latents for MOSS image training."""
import logging
import torch

from musubi_tuner import cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ImageDataset
from musubi_tuner.moss_image.vae import load_vae
from musubi_tuner.moss_image.cache import save_latents


@torch.no_grad()
def encode_and_save_batch(vae, batch):
    images = []
    for item in batch:
        content = item.content
        if isinstance(content, list):
            if len(content) != 1:
                raise ValueError("MOSS caching expects one image per item.")
            content = content[0]
        images.append(torch.from_numpy(content))
    pixels = torch.stack(images).permute(0, 3, 1, 2).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels / 127.5 - 1.0)
    for item, latent in zip(batch, latents):
        save_latents(item, latent.unsqueeze(1))


def setup_parser():
    parser = cache_latents.hv_setup_parser(cache_latents.setup_parser_common())
    parser.add_argument("--disable_mmap", action="store_true")
    return parser


def main():
    args = setup_parser().parse_args()
    if args.vae_dtype not in (None, "float32", "fp32"):
        raise ValueError("MOSS latent caching uses FP32 VAE encoding for numerical stability.")
    config = config_utils.load_user_config(args.dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(config, args, architecture=ARCHITECTURE_MOSS_IMAGE)
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets
    if any(not isinstance(dataset, ImageDataset) for dataset in datasets):
        raise ValueError("MOSS-Image requires image datasets, not video or audio datasets.")
    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width,
                                   args.console_back, args.console_num_images, fps=1)
        return
    if not args.vae:
        raise ValueError("Select the Flux.2 VAE for MOSS latent caching.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae = load_vae(args.vae, dtype=torch.float32, device=device, disable_mmap=args.disable_mmap)
    vae.eval().requires_grad_(False)
    cache_latents.encode_datasets(datasets, lambda batch: encode_and_save_batch(vae, batch), args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
