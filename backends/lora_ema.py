"""Run Musubi's post-hoc LoRA averaging from either GUI."""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from backends.numeric_settings import decimal_text


def plan_ema(checkpoints, output, decay="0.5"):
    paths = tuple(Path(line.strip().strip('"')).expanduser().resolve()
                  for line in str(checkpoints or "").splitlines() if line.strip())
    if len(set(paths)) != len(paths) or len(paths) < 2:
        raise ValueError("Choose at least two different LoRA checkpoints from the same run, oldest first.")
    if any(path.suffix.lower() != ".safetensors" or not path.is_file() for path in paths):
        raise ValueError("Each checkpoint must be an existing .safetensors file.")
    output = str(output or "").strip()
    destination = (Path(output).expanduser() if output else paths[-1].with_stem(paths[-1].stem + "_ema")).resolve()
    if destination.suffix.lower() != ".safetensors" or not destination.parent.is_dir() or destination.exists():
        raise ValueError("Choose a new .safetensors output in an existing folder.")
    try:
        beta = float(decimal_text(decay))
    except (ValueError, TypeError) as exc:
        raise ValueError("EMA decay must be a number from 0 to 1.") from exc
    if not 0 < beta < 1:
        raise ValueError("EMA decay must be greater than 0 and less than 1.")
    command = [sys.executable, "-m", "backends.lora_ema", "--output", str(destination), "--decay", str(beta),
               *(str(path) for path in paths)]
    return command, destination


def main():
    from safetensors import safe_open
    from musubi_tuner.lora_post_hoc_ema import merge_lora_weights_with_post_hoc_ema

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--decay", default="0.5")
    args = parser.parse_args()
    _, destination = plan_ema("\n".join(args.checkpoints), args.output, args.decay)
    known_metadata = {}
    for path in args.checkpoints:
        with safe_open(path, framework="pt") as checkpoint:
            metadata = checkpoint.metadata() or {}
            for key in ("ss_base_model_version", "ss_sd_model_hash", "ss_network_module"):
                value = metadata.get(key)
                if value and known_metadata.setdefault(key, value) != value:
                    raise ValueError(f"Checkpoint metadata conflicts for {key}: {path}")
    beta = float(decimal_text(args.decay))
    with TemporaryDirectory(prefix=".musubi-ema-", dir=destination.parent) as directory:
        staging = Path(directory) / "adapter.safetensors"
        merge_lora_weights_with_post_hoc_ema(args.checkpoints, True, beta, beta, None, str(staging))
        if os.name == "nt":
            staging.rename(destination)
        else:
            os.link(staging, destination)
    print(f"Saved to {destination}")


if __name__ == "__main__":
    main()
