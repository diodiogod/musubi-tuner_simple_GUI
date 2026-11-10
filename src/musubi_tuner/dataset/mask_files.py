"""Lightweight naming rules for mask sidecars shared by GUIs and datasets.

Fizgig and OneTrainer reserve ``photo-masklabel.png`` for photo's loss mask.
Do not guess from the word "mask": photos such as ``person_with_mask.jpg``
remain ordinary training images. This only excludes sidecars; it does not
enable masked-loss training or delete any files.
"""
from pathlib import Path

MASK_SUFFIX = "-masklabel"


def is_mask_stem(stem: str) -> bool:
    """Accept an extension-free media/cache item key."""
    return str(stem).casefold().endswith(MASK_SUFFIX)


def is_mask_image(path) -> bool:
    return is_mask_stem(Path(path).stem)
