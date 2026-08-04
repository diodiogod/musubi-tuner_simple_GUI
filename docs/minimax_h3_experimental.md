# MiniMax H3 image-only LoRA (experimental)

This fork contains an intentionally narrow MiniMax H3 training path for users with a 24 GB NVIDIA GPU. It trains a standard BF16 LoRA over Comfy's frozen, pruned ConvRot INT8 FL2VA transformer. It does **not** reconstruct or require the roughly 66 GB BF16 transformer.

The implementation is experimental and has completed CPU architecture, loader, forward, LoRA-gradient, cache-contract, backend, and GUI tests. It has also completed real one-step CUDA training at 1024x1024 with rank 16 on a 24 GB RTX 4090. This proves the narrow training path and memory target, but not long-run stability or training quality. Start with a short run and retain the original checkpoint.

## Required files

Download only the components needed for the phase you are running:

| Phase | File | Published size | Required? |
|---|---|---:|---|
| Train | [`minimax_h3_fl2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | 20.97 GB | Always |
| Cache captions | [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | 15.69 GB | Only when text caches must be created |
| Cache images | [`minimax_h3_video_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) | 5.21 GB | Only when latent caches must be created |

The files are used sequentially. The text encoder and VAE are not loaded during the LoRA training phase when their caches already exist. Existing compatible caches therefore let a user train with only the 20.97 GB pruned transformer available locally.

Do not select the Ref2VA checkpoint. This first implementation is FL2VA still-image training only.

## Safe first-run settings

Select **MiniMax H3 (Experimental)** in either GUI. The mode applies these defaults:

- LoRA rank and alpha: `16`
- mixed precision: `bf16`
- attention: `sdpa`
- gradient checkpointing: enabled
- timestep sampling: `krea2_shift`
- blocks to swap: `30`
- block-swap direction: H2D-only (enforced by the backend)
- ConvRot backward: `bf16`
- dataset batch size: `1`; use gradient accumulation for a larger effective batch

Image buckets must use dimensions divisible by 32. The latent and text cache commands use the same dataset TOML as training.

The `int8` ConvRot backward option is more experimental and needs working Triton kernels. Keep `bf16` for the validated baseline.

### Tuning block swap after the first run

Keep `30` swapped blocks for the safest first attempt on a 24 GB card. Once that configuration trains successfully, lowering the value can improve throughput by keeping more transformer blocks on the GPU, at the cost of higher VRAM use and less protection against a large bucket or temporary allocation spike.

| Swapped blocks | Current evidence | Guidance |
|---:|---|---|
| `30` | Completed the controlled 1024x1024 rank-16 optimizer-step validation at a 14,397 MiB physical peak | Safe automatic default |
| `15` | A real dataset run is training on an RTX 4090 at roughly 19–20 GB | Faster experimental option; monitor VRAM and return to `30` if it approaches the card limit or runs out of memory |

The `15`-block result is an in-progress field test, not a completed long-run validation. VRAM also varies with bucket dimensions, optimizer state, driver behavior, and other processes, so the GUI deliberately continues to select `30` automatically.

## CUDA validation result

The real published Comfy checkpoints were tested on an RTX 4090 selected by GPU UUID:

- 1024x1024 image, batch 1, rank/alpha 16
- 30 H2D-only swapped blocks with ring size 2
- BF16 ConvRot backward, SDPA, and gradient checkpointing
- one complete forward, backward, optimizer, intermediate-save, and final-save cycle
- finite loss (`0.218`) and 600/600 finite saved LoRA tensors
- all 200 initially-zero `lora_up` tensors became nonzero
- physical GPU peak: `14,397 MiB`, including about `1,279 MiB` present before launch

A smaller 256x256 rank-4 validation peaked at `12,978 MiB`. The compact NVFP4/AWQ text encoder and FP16 VAE also successfully produced their real cache files. These are smoke-test measurements, not a guarantee for every driver, optimizer, caption length, bucket, or future checkpoint revision.

## Deliberate limitations

- still images only; no video clips, reference media, or audio training
- LoRA only; no LoHa, LoKr, or full-model training
- batch size 1
- no in-training samples or standalone H3 generation
- no `torch.compile`, FP8-base conversion, or standard device-to-host block swap
- only the published pruned ConvRot INT8 FL2VA tensor contract is accepted

These restrictions keep the first version small, auditable, and suitable for later replacement by the final upstream interfaces. The loader validates every quantized projection, scale, marker, dtype, and model key before training instead of silently accepting a similar checkpoint.

## Why this is separate from upstream R1

[musubi-tuner PR #1018](https://github.com/kohya-ss/musubi-tuner/pull/1018) is a broader BF16 implementation with joint video/audio behavior. Its current R1 explicitly defers ConvRot, prequantized INT8, pruned AdaLN, and NVFP4/AWQ support. This fork reuses the compatible H3 architecture and cache contracts while isolating the experimental INT8 image path so it can later be replaced or reconciled without disturbing Wan, Flux.2, or Krea 2.

The still-image flow and compact-text-encoder behavior were also compared with [Fizgig v3.2.0](https://github.com/shootthesound/Fizgig/releases/tag/v3.2.0). Fizgig deliberately ships image-only, batch-1 H3 LoRA training without previews and reports its own 24 GB support as still being tested. This fork differs by loading the published pruned ConvRot INT8 transformer directly rather than downloading the BF16 transformer and converting it for training.

## First-run checklist

1. Close ComfyUI and other CUDA applications so the training GPU is empty.
2. Cache one small image and one short caption.
3. Set one training step, rank 4 or 8, batch size 1, `30` swapped blocks, and BF16 ConvRot backward.
4. Confirm the pruned checkpoint passes strict inspection without allocating a BF16 copy.
5. Confirm one optimizer step finishes, the loss is finite, the saved LoRA tensors are nonzero, and peak VRAM stays below the card limit.
6. Only then try rank 16 and a normal dataset.

If the one-step smoke fails in the loader or ConvRot backward on another setup, report the exact checkpoint, GPU, and traceback rather than weakening the checkpoint checks or falling back to a full BF16 download.
