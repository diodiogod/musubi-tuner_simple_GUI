"""Load optional Fizgig/OneTrainer masks aligned to the image training grid."""

import logging
from pathlib import Path

import cv2
import torch
from PIL import Image

from musubi_tuner.dataset.mask_files import MASK_SUFFIX
from musubi_tuner.dataset.media_utils import resize_image_to_bucket


logger = logging.getLogger(__name__)


def _packed_mask(path, original_size, bucket_size, pixels_per_token):
    with Image.open(path) as image:
        # A white drawing on a transparent canvas is a mask too.
        rgba = image.convert("RGBA")
        mask = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba).convert("L")
        if mask.size != original_size:
            mask = mask.resize(original_size, Image.Resampling.LANCZOS)
        cropped = resize_image_to_bucket(mask, bucket_size)
    width, height = bucket_size
    if min(width, height) < pixels_per_token or width % pixels_per_token or height % pixels_per_token:
        raise ValueError(f"Mask bucket {width}×{height} for {path} must be aligned to {pixels_per_token} pixels.")
    packed = cv2.resize(cropped, (width // pixels_per_token, height // pixels_per_token), interpolation=cv2.INTER_AREA)
    if not packed.any():
        raise ValueError(f"The mask has no visible training area in its bucket: {path}")
    return packed


def prepare_masks(datasets, pixels_per_token):
    count = 0
    for dataset in datasets:
        sources = sources_by_stem(dataset)
        manager = dataset.batch_manager
        packed_masks = {}
        for bucket in manager.buckets.values():
            # Repeated training items share one mask.
            for item in {item.latent_cache_path: item for item in bucket}.values():
                images = sources.get(item.item_key, ())
                masks = [image.with_name(image.stem + MASK_SUFFIX + ".png") for image in images]
                masks = [mask for mask in masks if mask.is_file()]
                if not masks:
                    continue
                if len(images) != 1:
                    raise ValueError(f"Cannot match the mask for {item.item_key}: multiple source images share its filename stem.")
                packed_masks[item.latent_cache_path] = _packed_mask(masks[0], item.original_size, item.bucket_size[:2], pixels_per_token)
        manager.image_loss_masks = packed_masks
        count += len(packed_masks)
    if not count:
        raise ValueError("Masked training found no matching photo-masklabel.png files beside the cached training images.")
    logger.info("Masked loss: %d cached images have masks. Images without a mask use the whole image.", count)


def mean_image_loss(loss, mask, background_weight):
    """Reduce each still photo over its weighted region, then average photos."""
    if mask is None:
        return loss.mean()
    if (loss.ndim not in {4, 5} or (loss.ndim == 5 and loss.shape[2] != 1)
            or mask.shape != (loss.shape[0], *loss.shape[-2:])):
        raise ValueError("The training mask grid does not match the still-image prediction. Check the image and latent cache.")
    foreground = mask.to(device=loss.device, dtype=torch.float32) / 255
    weights = foreground.lerp(torch.ones_like(foreground), background_weight)
    spatial_loss = loss.float().flatten(1, -3).mean(dim=1)
    return ((spatial_loss * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2))).mean()


def image_mse_loss(prediction, target, mask, background_weight):
    if mask is None:
        return torch.nn.functional.mse_loss(prediction, target)
    loss = torch.nn.functional.mse_loss(prediction, target, reduction="none")
    return mean_image_loss(loss, mask, background_weight)


def sources_by_stem(dataset):
    source = dataset.datasource
    if dataset.image_jsonl_file:
        image_paths = (row.get("image_path") or row.get("image_path_0") for row in source.data)
    else:
        image_paths = source.image_paths
    sources = {}
    for path in filter(None, image_paths):
        image = Path(path)
        sources.setdefault(image.stem, set()).add(image)
    return sources
