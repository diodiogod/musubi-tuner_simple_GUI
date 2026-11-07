"""Frozen sampling adapters, composed live so training weights are never merged."""
from contextlib import contextmanager
import math

import torch
from safetensors.torch import load_file
from musubi_tuner.networks import lora_moss_image


def normalize_lora_keys(state):
    """Accept Musubi and ComfyUI/PEFT MOSS main-block LoRA names."""
    result = {}
    for key, value in state.items():
        original = key
        for prefix in ("model.diffusion_model.", "diffusion_model.", "base_model.model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        key = key.replace(".lora_A.weight", ".lora_down.weight").replace(".lora_B.weight", ".lora_up.weight")
        for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if key.endswith(suffix) and not key.startswith("lora_unet_"):
                key = "lora_unet_" + key[:-len(suffix)].replace(".", "_") + suffix
                break
        if key in result:
            raise ValueError(f"Duplicate MOSS adapter key: {original}")
        result[key] = value
    return result


def load_sampling_adapter(path, multiplier, model, device, dtype=torch.bfloat16):
    if not math.isfinite(multiplier):
        raise ValueError("Sampling LoRA strength must be finite.")
    state = normalize_lora_keys(load_file(str(path), device="cpu"))
    if not any(key.endswith(".lora_down.weight") for key in state):
        raise ValueError("The sampling adapter must be a MOSS LoRA with down/up weights.")
    network = lora_moss_image.create_arch_network_from_weights(multiplier, state, unet=model, for_inference=True)
    # Validate before attaching any forward hooks: an incompatible adapter must
    # not partially alter the running training model.
    expected = {f"{module.lora_name}.{key}": value for module in network.unet_loras
                for key, value in module.state_dict().items() if not key.startswith("org_module.")}
    for key, value in expected.items():
        if key.endswith(".alpha"):
            state.setdefault(key, value)
    if set(state) != set(expected) or any(state[key].shape != value.shape for key, value in expected.items() if key in state):
        raise ValueError("Sampling LoRA contains incompatible MOSS layers or tensor shapes.")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.load_state_dict(state, strict=True)
    network.to(device=device, dtype=dtype).eval().requires_grad_(False)
    network.set_enabled(False)
    return network


@contextmanager
def sampling_adapter_enabled(network, device=None, offload=False):
    try:
        if network is not None:
            if device is not None:
                network.to(device)
            network.set_enabled(True)
        yield
    finally:
        if network is not None:
            network.set_enabled(False)
            if offload:
                network.to("cpu")
