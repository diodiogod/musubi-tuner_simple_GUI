"""Naming helpers for repeated training continuations."""

import re


def split_dynamic_suffix(name, suffix):
    """Strip repeated ``-suffix`` tokens and return their combined generation."""
    base_name = str(name or "").strip() or "training"
    generation = 0
    pattern = re.compile(rf"-{re.escape(suffix)}(?P<generation>\d*)$", re.IGNORECASE)
    while True:
        match = pattern.search(base_name)
        if match is None:
            break
        generation += int(match.group("generation") or 1)
        base_name = base_name[:match.start()].rstrip("-") or "training"
    return base_name, generation


def dynamic_suffix_name(name, suffix, generation=None):
    """Build a canonical numbered suffix without accumulating suffix chains."""
    base_name, current_generation = split_dynamic_suffix(name, suffix)
    next_generation = current_generation + 1 if generation is None else max(1, int(generation))
    numbered_suffix = f"-{suffix}" if next_generation == 1 else f"-{suffix}{next_generation}"
    return f"{base_name}{numbered_suffix}"


def split_continuation_name(name):
    """Return the original run name and its inferred continuation generation."""
    return split_dynamic_suffix(name, "cont")


def continuation_name(name, generation=None):
    """Build a canonical continuation name without accumulating suffix chains."""
    return dynamic_suffix_name(name, "cont", generation)
