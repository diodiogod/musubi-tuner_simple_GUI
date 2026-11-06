"""REPA validation shared with GUI preflight, without loading the ML runtime."""
import math


def validate_config(args, num_layers=42):
    if not getattr(args, "moss_repa_enabled", False):
        return
    if not math.isfinite(args.moss_repa_weight) or args.moss_repa_weight <= 0:
        raise ValueError("REPA strength must be finite and greater than zero.")
    if not 0 <= args.moss_repa_block < num_layers:
        raise ValueError(f"REPA block must be between 0 and {num_layers - 1} (counting from zero).")
    if args.moss_repa_image_size < 14 or args.moss_repa_image_size % 14:
        raise ValueError("REPA image size must be a positive multiple of 14, such as 224 or 518.")
    if args.moss_repa_cache_size < 0:
        raise ValueError("REPA memory-cache size cannot be negative.")
    if not str(args.moss_repa_encoder).strip():
        raise ValueError("Select a DINOv2 model ID or local model folder for REPA.")
    if args.moss_repa_device not in {"training", "cpu"}:
        raise ValueError("REPA compute device must be training or cpu.")
    if getattr(args, "mixed_precision", "bf16") == "fp16":
        raise ValueError("Experimental MOSS REPA supports BF16 or no mixed precision; choose BF16.")
