"""Sampling cadence helpers shared by the Modern GUI and command adapters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def estimate_steps_per_epoch(
    dataset_config: str,
    gradient_accumulation_steps: str | int | float = 1,
    text: str | None = None,
) -> dict[str, Any]:
    """Estimate optimizer steps in one epoch using Musubi-visible media.

    The estimate follows the GUI's dataset audit: usable media multiplied by
    repeats, grouped by each source's batch size, then divided by gradient
    accumulation. Bucket packing and distributed launch details can make the
    final value differ by a step, so callers should label this as an estimate.
    """

    from modern_gui.dataset_documents import inspect_dataset_sources, summarize_document

    path = Path(str(dataset_config or "")).expanduser()
    if text is None:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset TOML does not exist: {path}")
        text = path.read_text(encoding="utf-8-sig")
    summary = summarize_document(text, str(path))
    audit = inspect_dataset_sources(text, str(path))
    reports = {int(report["index"]): report for report in audit.get("datasets", [])}
    batches = 0
    effective_samples = 0
    sources: list[dict[str, Any]] = []
    for dataset in summary.get("datasets", []):
        index = int(dataset.get("index", len(sources)))
        report = reports.get(index, {})
        if report.get("error"):
            raise ValueError(f"Source {index + 1}: {report['error']}")
        usable = int(report.get("trainer_usable_count", 0))
        repeats = max(1, int(dataset.get("repeats") or 1))
        batch_size = max(1, int(dataset.get("batch_size") or 1))
        samples = usable * repeats
        source_batches = math.ceil(samples / batch_size) if samples else 0
        effective_samples += samples
        batches += source_batches
        sources.append(
            {
                "index": index,
                "usable_samples": usable,
                "repeats": repeats,
                "batch_size": batch_size,
                "effective_samples": samples,
                "batches": source_batches,
            }
        )
    try:
        accumulation = max(1, int(float(str(gradient_accumulation_steps or 1))))
    except (TypeError, ValueError):
        accumulation = 1
    steps = math.ceil(batches / accumulation) if batches else 0
    return {
        "steps_per_epoch": steps,
        "batches_per_epoch": batches,
        "effective_samples": effective_samples,
        "gradient_accumulation_steps": accumulation,
        "datasets": sources,
        "estimated": True,
    }


def fractional_epoch_to_steps(
    cadence: str | int | float,
    dataset_config: str,
    gradient_accumulation_steps: str | int | float = 1,
) -> int:
    """Convert a fractional epoch cadence (for example 0.5) to steps."""

    value = float(str(cadence).strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Epoch sampling cadence must be greater than zero.")
    estimate = estimate_steps_per_epoch(dataset_config, gradient_accumulation_steps)
    if estimate["steps_per_epoch"] <= 0:
        raise ValueError("The dataset has no usable samples, so fractional epoch sampling cannot be converted to steps.")
    return max(1, round(estimate["steps_per_epoch"] * value))
