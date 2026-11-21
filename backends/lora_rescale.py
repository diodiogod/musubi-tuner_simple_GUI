"""Bake a ComfyUI strength into a copy of a plain additive LoRA."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from backends.numeric_settings import decimal_text


@dataclass(frozen=True)
class RescalePlan:
    source: Path
    output: Path
    current: float
    target: float

    def command(self) -> list[str]:
        return [
            sys.executable, "-m", "backends.lora_rescale",
            "--input", str(self.source), "--output", str(self.output),
            f"--current-strength={self.current}", f"--target-strength={self.target}",
        ]


def _strength(value, label: str) -> float:
    try:
        if not isinstance(value, bool):
            strength = float(decimal_text(value))
            if math.isfinite(strength) and strength != 0:
                return strength
    except (TypeError, ValueError, OverflowError):
        pass
    raise ValueError(f"{label} must be a finite, nonzero number.")


def plan_rescale(source_path, output_path, current_strength, target_strength=1.0) -> RescalePlan:
    current = _strength(current_strength, "Current strength")
    target = _strength(target_strength, "Target strength")
    ratio = current / target
    if not math.isfinite(ratio) or ratio == 0:
        raise ValueError("The strength ratio is outside the supported numeric range.")

    source = Path(str(source_path or "").strip().strip('"')).expanduser().absolute()
    if source.suffix.lower() != ".safetensors" or not source.is_file():
        raise ValueError("Choose an existing LoRA .safetensors file.")
    output_name = str(output_path or "").strip().strip('"') or f"{source.stem}_rescaled.safetensors"
    output = source.parent / Path(output_name).expanduser()
    if output.suffix.lower() != ".safetensors":
        raise ValueError("The output filename must end in .safetensors.")
    if not output.parent.is_dir():
        raise ValueError("Choose an existing folder for the rescaled LoRA.")
    output = output.parent.resolve() / output.name
    if output == source.resolve():
        raise ValueError("Choose a different filename for the rescaled LoRA.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"The output already exists. Choose a new filename.\n{output}")
    return RescalePlan(source, output, current, target)


_WEIGHT_PAIRS = (
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_A.default.weight", ".lora_B.default.weight"),
)


def _rescaled_alphas(tensors, plan: RescalePlan):
    import torch

    alphas = {}
    recognized = set()
    for down_suffix, up_suffix in _WEIGHT_PAIRS:
        for down_key, down in tensors.items():
            if not down_key.endswith(down_suffix):
                continue
            module = down_key.removesuffix(down_suffix)
            up_key, alpha_key = module + up_suffix, module + ".alpha"
            up = tensors.get(up_key)
            if alpha_key in alphas:
                raise ValueError(f"Multiple LoRA weight layouts for {module}.")
            if (up is None or down.ndim < 2 or up.ndim < 2 or not down.numel() or not up.numel()
                    or down.shape[0] != up.shape[1] or not down.is_floating_point() or not up.is_floating_point()):
                raise ValueError(f"Missing or incompatible LoRA weight pair for {module}.")

            rank = down.shape[0]
            alpha = tensors.get(alpha_key)
            if alpha is not None and alpha.numel() != 1:
                raise ValueError(f"Expected a scalar LoRA alpha for {module}.")
            # Match ComfyUI's alpha/rank and strength multiplication order.
            # An absent alpha means a normalized scale of 1. Keep the learned
            # matrices in their original precision, including BF16 and FP16.
            normalized = 1.0 if alpha is None else alpha.item() / rank
            effective = plan.current * normalized
            scaled = (effective / plan.target) * rank
            if not math.isfinite(scaled) or (scaled == 0 and normalized != 0):
                raise ValueError(f"Rescaled alpha for {module} is outside the supported numeric range.")
            alphas[alpha_key] = torch.tensor(scaled, dtype=torch.float64)
            recognized.update((down_key, up_key, alpha_key))

    if not alphas:
        raise ValueError("Choose a standard additive LoRA with up/down or A/B weight pairs.")
    unsupported = tensors.keys() - recognized
    if unsupported:
        raise ValueError(f"This LoRA uses an unsupported tensor: {min(unsupported)}. Only plain additive LoRAs are supported.")
    return alphas


def rescale_lora(plan: RescalePlan) -> int:
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(plan.source, framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        tensors = {key: reader.get_tensor(key) for key in reader.keys()}
    alphas = _rescaled_alphas(tensors, plan)
    tensors.update(alphas)

    if "sshs_model_hash" in metadata or "sshs_legacy_hash" in metadata:
        from musubi_tuner.utils.model_utils import precalculate_safetensors_hashes

        metadata["sshs_model_hash"], metadata["sshs_legacy_hash"] = precalculate_safetensors_hashes(tensors, metadata)
    if "modelspec.hash_sha256" in metadata:
        from musubi_tuner.utils.sai_model_spec import precalculate_safetensors_hashes

        metadata["modelspec.hash_sha256"] = precalculate_safetensors_hashes(tensors)

    # Publish only a complete file and refuse to replace a destination created
    # after validation, even if another GUI or process is writing concurrently.
    with NamedTemporaryFile(dir=plan.output.parent, prefix=".lora-rescale-", suffix=".tmp", delete=False) as temp:
        staging = Path(temp.name)
    try:
        save_file(tensors, staging, metadata=metadata or None)
        if os.name == "nt":
            staging.rename(plan.output)  # Windows rename fails if the destination exists.
        else:
            os.link(staging, plan.output)
    finally:
        staging.unlink(missing_ok=True)
    return len(alphas)


def main() -> None:
    from safetensors import SafetensorError

    parser = argparse.ArgumentParser(description="Save a standard LoRA with a new equivalent ComfyUI strength.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="", help="Defaults to <input>_rescaled.safetensors beside the input.")
    parser.add_argument("--current-strength", required=True)
    parser.add_argument("--target-strength", default="1.0")
    args = parser.parse_args()
    try:
        plan = plan_rescale(args.input, args.output, args.current_strength, args.target_strength)
        layers = rescale_lora(plan)
    except (OSError, ValueError, SafetensorError) as exc:
        parser.exit(1, f"LoRA rescaling failed. {exc}\n")
    print(f"Rescaled {layers} LoRA layers from strength {plan.current:g} to {plan.target:g}.")
    print(f"Saved to {plan.output}")


if __name__ == "__main__":
    main()
