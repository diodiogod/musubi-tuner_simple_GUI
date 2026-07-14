"""Pure helpers for organizing generated training samples in the desktop GUI."""

import re
from pathlib import Path


_TRAINING_SAMPLE_NAME_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<marker>e?\d{6})_(?P<prompt>\d{2})_"
    r"(?P<timestamp>\d{14})(?:_(?P<seed>-?\d+))?(?P<tail>.*)$",
    re.IGNORECASE,
)


def parse_training_sample_path(fpath):
    """Return stable comparison metadata for a Musubi training sample path."""
    path = Path(fpath)
    match = _TRAINING_SAMPLE_NAME_RE.match(path.stem)
    if match is None:
        return None

    marker = match.group("marker").lower()
    is_epoch = marker.startswith("e")
    sequence = int(marker[1:] if is_epoch else marker)
    prefix = match.group("prefix").rstrip("_")
    prompt_index = int(match.group("prompt"))
    seed = match.group("seed")
    tail = match.group("tail") or ""
    return {
        "group_key": (
            str(path.parent).casefold(), prefix.casefold(), prompt_index,
            seed, tail.casefold(), path.suffix.casefold(),
        ),
        "prefix": prefix,
        "prompt_index": prompt_index,
        "seed": seed,
        "sequence": sequence,
        "sequence_kind": "epoch" if is_epoch else "step",
        "sequence_label": f"Epoch {sequence}" if is_epoch else f"Step {sequence}",
    }
