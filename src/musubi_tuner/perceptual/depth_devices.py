"""Device selection helpers shared by experimental differentiable depth losses."""

from __future__ import annotations

import torch


def resolve_depth_vae_device(spec: str, training_device: torch.device) -> torch.device:
    """Resolve ``training`` or ``secondary`` to a visible logical CUDA device."""

    training_device = torch.device(training_device)
    value = str(spec or "training").strip().lower()
    if value == "training":
        return training_device
    if training_device.type != "cuda":
        raise ValueError("A secondary depth VAE GPU requires CUDA training")
    training_index = training_device.index
    if training_index is None:
        training_index = torch.cuda.current_device()
    if value == "secondary":
        candidates = [index for index in range(torch.cuda.device_count()) if index != training_index]
        if not candidates:
            raise ValueError(
                "Secondary depth VAE GPU was selected, but PyTorch can see only one CUDA device. "
                "Make both GPUs visible to the training process or use the training GPU setting."
            )
        return torch.device("cuda", candidates[0])
    try:
        requested = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Depth VAE device must be 'training', 'secondary', or a CUDA device such as 'cuda:1'") from exc
    if requested.type != "cuda" or requested.index is None or requested.index >= torch.cuda.device_count():
        raise ValueError(f"Depth VAE device {value!r} is not an available logical CUDA device")
    return requested
