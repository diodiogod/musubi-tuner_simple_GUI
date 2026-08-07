"""Frozen MiniMax-H3 training-assistant and sparse base-preservation helpers.

The assistant mechanism follows Ostris AI Toolkit's live, unmerged training
adapter design.  The published alpha adapter remains active while the user's
LoRA trains, is disabled for samples, and is never written into user outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file


DEFAULT_ASSISTANT = (
    "ostris/minimax_h3_training_adapter/"
    "minimax_h3_training_adapter_v1.safetensors"
)


def resolve_assistant_path(value: str) -> Path:
    candidate = Path(str(value).strip()).expanduser()
    if candidate.is_file():
        return candidate
    parts = str(value).strip().replace("\\", "/").split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "H3 assistant must be an existing .safetensors file or a "
            "Hugging Face user/repository/filename path."
        )
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id="/".join(parts[:2]), filename=parts[2]))


def convert_ai_toolkit_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert Ostris/diffusers A/B names into Musubi's native LoRA names."""

    converted: dict[str, torch.Tensor] = {}
    module_names: set[str] = set()
    prefix = "diffusion_model."
    for key, tensor in weights.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.endswith(".lora_A.weight"):
            module_path = suffix[: -len(".lora_A.weight")]
            weight_name = "lora_down.weight"
        elif suffix.endswith(".lora_B.weight"):
            module_path = suffix[: -len(".lora_B.weight")]
            weight_name = "lora_up.weight"
        else:
            continue
        native_name = "lora_unet_" + module_path.replace(".", "_")
        converted[f"{native_name}.{weight_name}"] = tensor
        module_names.add(native_name)
    if not module_names:
        raise ValueError("The selected file contains no compatible MiniMax-H3 assistant LoRA weights.")
    for name in module_names:
        down = converted.get(f"{name}.lora_down.weight")
        up = converted.get(f"{name}.lora_up.weight")
        if down is None or up is None:
            raise ValueError(f"MiniMax-H3 assistant module is incomplete: {name}")
        converted[f"{name}.alpha"] = torch.tensor(float(down.shape[0]))
    return converted


def load_live_assistant(transformer, source: str, device: torch.device, dtype: torch.dtype):
    from musubi_tuner.networks import lora

    path = resolve_assistant_path(source)
    weights = convert_ai_toolkit_weights(load_file(str(path), device="cpu"))
    network = lora.create_network_from_weights(
        ["DiTBlock", "RefinerBlock"],
        1.0,
        weights,
        unet=transformer,
        for_inference=True,
    )
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise ValueError(
            "MiniMax-H3 assistant does not exactly match this model: "
            f"missing={info.missing_keys[:8]}, unexpected={info.unexpected_keys[:8]}"
        )
    network.to(device=device, dtype=dtype).requires_grad_(False).eval()
    network.set_enabled(True)
    network.assistant_source = os.fspath(path)
    return network


def should_preserve_base(global_step: int, every_n_steps: int) -> bool:
    if every_n_steps < 1:
        raise ValueError("H3 base-preservation cadence must be at least 1")
    return (int(global_step) + 1) % int(every_n_steps) == 0


def base_preservation_loss(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(prediction.float(), reference.detach().float())
