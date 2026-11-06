"""MOSS cache serialization using the shared Musubi dataset contract."""
import torch

from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE_FULL
from musubi_tuner.dataset.cache_io import save_latent_cache_common, save_text_encoder_output_cache_common
from musubi_tuner.utils.model_utils import dtype_to_str


def save_latents(item, latent):
    if latent.ndim != 4 or tuple(latent.shape[:2]) != (128, 1):
        raise ValueError("MOSS requires normalized, packed image latents [128,1,H,W].")
    _, frames, height, width = latent.shape
    key = f"latents_{frames}x{height}x{width}_{dtype_to_str(latent.dtype)}"
    save_latent_cache_common(item, {key: latent.detach().cpu().contiguous()}, ARCHITECTURE_MOSS_IMAGE_FULL)


def save_embeddings(item, embed, dop_embed=None, dop_signature=None):
    tensors = {}
    for name, value in (("moss_embed", embed), ("dop_moss_embed", dop_embed)):
        if value is None:
            continue
        if value.ndim != 2 or value.shape[1] != 4096 or value.shape[0] < 1:
            raise ValueError("MOSS text embeddings must have shape [tokens,4096].")
        tensors[f"varlen_{name}_{dtype_to_str(value.dtype)}"] = value.detach().cpu().contiguous()
    if dop_signature is not None:
        tensors["dop_signature"] = dop_signature.detach().cpu()
    save_text_encoder_output_cache_common(item, tensors, ARCHITECTURE_MOSS_IMAGE_FULL)
