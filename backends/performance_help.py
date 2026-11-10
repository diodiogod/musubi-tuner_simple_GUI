"""Short, field-specific guidance shared by the training interfaces."""

DETAILS = {
    "mixed_precision": (
        "Controls the number format used during model calculations. BF16 uses less memory than FP32 and is a good "
        "starting choice for your RTX 4090. Trainable adapter weights can still stay in FP32. This setting does "
        "not change image resolution or choose an optimizer. Keep BF16 for MOSS; full-weight BF16 is a separate option."
    ),
    "attention_mechanism": (
        "Chooses the implementation of attention, a major calculation inside the model. Start with SDPA: PyTorch "
        "chooses a compatible kernel automatically. Other choices require compatible optional packages. MOSS training "
        "currently requires SDPA, even if SageAttention is available for previews. This setting does not enable an "
        "extra training technique or change your captions."
    ),
    "gradient_checkpointing": (
        "Saves GPU memory by recalculating some intermediate results during backward instead of keeping them all. "
        "Start with it ON on a 24 GB card. OFF can be faster when the entire training step fits comfortably. "
        "This is unrelated to saved checkpoint files: turning it off does not disable saving or resuming training."
    ),
    "blocks_to_swap": (
        "Chooses how many model blocks move between GPU memory and system RAM during each step. More swapped "
        "blocks can make a larger workload fit, but transfers usually slow training. Zero keeps all blocks on the GPU "
        "and is fastest when the step fits. MOSS compilation requires zero swapped blocks; turn Compile off if you "
        "need swapping. Swapping does not deliberately reduce training quality."
    ),
    "compile": (
        "Builds optimized versions of the model blocks to reduce repeated calculation overhead. Longer runs can "
        "be faster afterward, but the first step and new image/caption shapes may pause while compiling. "
        "For MOSS, use the optional performance runtime, zero swapped blocks, and ordinary GPU gradient checkpointing. "
        "Leave it off when you need block swapping or a simple troubleshooting baseline. It does not change the "
        "optimizer, noise schedule, or dataset."
    ),
    "persistent_data_loader_workers": (
        "Keeps the CPU processes that read training data alive between epochs, avoiding the cost of restarting them. "
        "It only matters when Data loader workers is greater than zero. Turn it off if you need those processes and "
        "their RAM released between epochs. It does not keep model weights or depth helpers on the GPU."
    ),
    "max_data_loader_n_workers": (
        "Limits the CPU processes used to read cached training data. Start with 1 or 2 on Windows; zero loads data "
        "in the training process and can help troubleshoot worker errors. More workers can reduce waiting for data, "
        "but also use more RAM and do not guarantee faster steps. This is unrelated to GPU block swapping."
    ),
    "fp8_base": (
        "Loads supported frozen base-model weights in an 8-bit floating-point format to reduce their memory use. "
        "Support depends on the model and GPU. This does not make the trainable LoRA an 8-bit adapter. "
        "MOSS does not support this FP8 option: keep it off and use its BF16 or ConvRot INT8 checkpoint instead."
    ),
    "fp8_scaled": (
        "Adds scale information to the supported FP8 base-weight path to reduce quantization error. It belongs to "
        "FP8 loading and is not an extra quality-preservation loss. Follow the selected model's FP8 requirements. "
        "For MOSS, keep both FP8 options off; its ConvRot INT8 support is selected separately."
    ),
    "moss_fast_lora": (
        "Uses a lighter calculation for an ordinary LoRA to save GPU memory and time. Your learned weights keep "
        "their normal FP32 precision; only the calculation uses BF16. Try ON for ordinary MOSS LoRA. "
        "Keep it OFF for LoKr or LoHa: they use different adapter calculations. Dropout uses the ordinary path. "
        "This is not full-BF16 training and needs no recaching. Small rounding differences are possible."
    ),
    "moss_unpadded_batches": (
        "Runs each image with its own caption length instead of filling shorter captions with blank positions. "
        "This follows Fizgig's current MOSS approach and can avoid unnecessary attention work. Try ON, with "
        "Caption padding set to zero. Images in a batch still contribute to the same optimizer update. "
        "It does not crop your images, shorten captions, or require rebuilding their cache."
    ),
    "moss_text_pad_multiple": (
        "Rounds caption lengths up to a small multiple by adding masked blank positions. This can reduce the "
        "number of different shapes the compiler has to prepare, but the extra positions also cost work. "
        "Use 0 with Unpadded image passes. As a separate experiment with masked batches, try 16. "
        "This is not a caption-length limit: it neither removes words nor changes the text cache."
    ),
    "moss_compile_checkpoint": (
        "Balances speed and GPU memory use when compilation and gradient checkpointing work together. "
        "Start with Inside for speed. Try Outside if higher-resolution training runs out of GPU memory; "
        "it arranges the repeated calculations differently and may use less memory. This only matters when Compile "
        "and gradient checkpointing are enabled. It does not control saving or resuming checkpoints."
    ),
    "moss_sdpa_backend": (
        "Chooses whether to leave attention-kernel selection to PyTorch or prefer cuDNN. Start with Default. "
        "cuDNN preferred is an alternative worth timing; planning new shapes can take longer initially, and "
        "unsupported shapes can use another available kernel. This only affects SDPA calculations. "
        "It does not install SageAttention, change captions, or add a quality-preservation pass."
    ),
}

MOSS_KEYS = tuple(key for key in DETAILS if key.startswith("moss_"))
