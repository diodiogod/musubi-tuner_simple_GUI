import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open


def read_header(path: Path):
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
    header = json.loads(header_bytes.decode("utf-8"))
    return header_size, header_bytes, header


def tensor_bytes(t: torch.Tensor) -> bytes:
    if t.dim() == 0:
        t = t.unsqueeze(0)
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description="Merge Krea 2 txtfusion.projector diff into a base checkpoint.")
    parser.add_argument("--base", required=True, help="Base Krea 2 checkpoint (.safetensors)")
    parser.add_argument("--patch", required=True, help="Patch file with diffusion_model.txtfusion.projector.diff")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--strength", type=float, default=1.0, help="Multiplier for the diff tensor")
    args = parser.parse_args()

    base_path = Path(args.base)
    patch_path = Path(args.patch)
    output_path = Path(args.output)

    header_size, header_bytes, header = read_header(base_path)
    target_key = "txtfusion.projector.weight"
    patch_key = "diffusion_model.txtfusion.projector.diff"

    if target_key not in header:
        raise KeyError(f"{target_key} not found in base checkpoint")

    with safe_open(str(base_path), framework="pt", device="cpu") as f:
        base_tensor = f.get_tensor(target_key)
    with safe_open(str(patch_path), framework="pt", device="cpu") as f:
        patch_tensor = f.get_tensor(patch_key)

    merged = (base_tensor.to(torch.float32) + args.strength * patch_tensor.to(torch.float32)).to(base_tensor.dtype)
    merged_bytes = tensor_bytes(merged)

    offset_start, offset_end = header[target_key]["data_offsets"]
    if len(merged_bytes) != (offset_end - offset_start):
        raise ValueError("Merged tensor byte size does not match original tensor byte size")

    absolute_start = 8 + header_size + offset_start
    absolute_end = 8 + header_size + offset_end

    chunk_size = 8 * 1024 * 1024
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with base_path.open("rb") as src, output_path.open("wb") as dst:
        pos = 0
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break

            chunk_start = pos
            chunk_end = pos + len(chunk)

            if chunk_end <= absolute_start or chunk_start >= absolute_end:
                dst.write(chunk)
            else:
                mutable = bytearray(chunk)
                patch_from = max(absolute_start, chunk_start)
                patch_to = min(absolute_end, chunk_end)
                local_start = patch_from - chunk_start
                local_end = patch_to - chunk_start
                merged_start = patch_from - absolute_start
                merged_end = patch_to - absolute_start
                mutable[local_start:local_end] = merged_bytes[merged_start:merged_end]
                dst.write(mutable)

            pos += len(chunk)

    print(f"Wrote merged checkpoint: {output_path}")
    print(f"Target key: {target_key}")
    print(f"Strength: {args.strength}")
    print(f"Original: {base_tensor.flatten().tolist()}")
    print(f"Patch: {patch_tensor.flatten().tolist()}")
    print(f"Merged: {merged.flatten().tolist()}")


if __name__ == "__main__":
    main()
