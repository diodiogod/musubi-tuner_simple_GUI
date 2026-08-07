# MiniMax H3 image-only LoRA (experimental)

This fork contains an intentionally narrow MiniMax H3 training path for users with a 24 GB NVIDIA GPU. It trains a standard BF16 LoRA over Comfy's frozen, pruned ConvRot INT8 FL2VA transformer. It does **not** reconstruct or require the roughly 66 GB BF16 transformer.

The implementation is experimental and has completed CPU architecture, loader, forward, LoRA-gradient, cache-contract, backend, sampler, and GUI tests. A real 1024x1024 rank-16 run completed two epochs on a 24 GB RTX 4090, and its LoRA produced good subject resemblance in inference. The same card has also completed compact standalone inference, a scheduled preview followed by training, two-step DOP plus weight-noise training, a differentiable depth-anchor step, and the H3 DRaFT generation/backward/save path. These are focused smoke tests, not validation of every long-run recipe. Start new features with a short run and retain the original checkpoint.

## Required files

Download only the components needed for the phase you are running:

| Phase | File | Published size | Required? |
|---|---|---:|---|
| Train | [`minimax_h3_fl2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | 20.97 GB | Always |
| Captions / previews / face refinement | [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | 15.69 GB | Caption caching or generation only |
| Images / previews / depth / face refinement | [`minimax_h3_video_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) | 5.21 GB | Latent caching or differentiable/image decoding only |

For ordinary LoRA training the files are used sequentially. The text encoder and VAE are not loaded in the training loop when compatible caches exist and previews/depth are off. Existing compatible caches therefore let a user train with only the 20.97 GB pruned transformer available locally. Scheduled previews, depth anchoring, and face refinement deliberately load the required helper at their own phase.

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
- H3 quality protection: enabled, sigma-balanced contrastive target strength `4.0`
- dataset batch size: `1`; use gradient accumulation for a larger effective batch

Image buckets must use dimensions divisible by 32. The latent and text cache commands use the same dataset TOML as training.

**Text Cache Precision** controls only the saved caption embeddings. The corrected Comfy-style Qwen3-VL tower continues to compute in FP32; the recommended `bfloat16` option converts its final layer-50 states before saving, halves their cache footprint, and reduces training-time cache traffic. Changing this option requires only **Rebuild Caption/Text Cache**. Image latents do not need to be rebuilt. `float32` remains available for controlled comparisons. Dynamic Sigma also stores one empty-prompt embedding in each caption cache; rebuild only this cache once after enabling that layer. The assistant and drift check do not require the empty-prompt entry by themselves.

MiniMax H3 VAE posterior sampling is forced to FP32 even though the published checkpoint filename says `fp16`. New caches record `posterior_policy=video_vae=fp32`; **Rebuild Image/Latent Cache** replaces older unmarked experimental caches instead of silently reusing them. Preview and depth decoding use FP16, matching the released decoder's practical path.

The `int8` ConvRot backward option is more experimental and needs working Triton kernels. Keep `bf16` for the validated baseline.

### Tuning block swap after the first run

Keep `30` swapped blocks for the safest first attempt on a 24 GB card. Once that configuration trains successfully, lowering the value can improve throughput by keeping more transformer blocks on the GPU, at the cost of higher VRAM use and less protection against a large bucket or temporary allocation spike.

| Swapped blocks | Current evidence | Guidance |
|---:|---|---|
| `30` | Completed the controlled 1024x1024 rank-16 optimizer-step validation at a 14,397 MiB physical peak | Safe automatic default |
| `15` | Completed a real two-epoch dataset run at roughly 19–20 GB; the resulting LoRA worked with good resemblance | Faster validated-on-one-machine option; monitor VRAM and return to `30` if it approaches the card limit or runs out of memory |

The `15`-block result is a completed field test, but VRAM varies with bucket dimensions, optimizer state, driver behavior, regularizers, previews, and other processes. The GUI deliberately continues to select `30` automatically. Depth anchoring should start at 30 or more; face refinement starts at 35.

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

## Preview and inference

Both GUIs can now generate a standalone MiniMax H3 image from an individual sample-prompt card and automatically use the selected or newest run LoRA. The command loads the compact text encoder first, unloads it, denoises with the pruned ConvRot INT8 DiT, unloads the DiT, and finally loads the VAE. This sequential design avoids requiring all three large components in VRAM together.

Scheduled in-training image previews are also supported. Recommended prompt settings are 768x768, 28 steps, shift 12, one frame, guidance/CFG 1.0, and no negative prompt. Before VAE decode the trainer can temporarily park enough frozen resident DiT weights on CPU to reach its preview free-VRAM target, then restores them before training continues. A 256x256 two-step preview was generated at step 0 and the following optimizer step completed successfully on a 24 GB RTX 4090. Begin a real run with one prompt and a generous cadence because normal preview sizes and step counts are substantially heavier.

The standalone CLI is `src/musubi_tuner/minimax_h3_image_generate.py`. Its decoder duplicates the single image latent into two temporal tokens and keeps the first decoded frame; this avoids the released video VAE's visibly poor single-token shortcut.

## Advanced regularization and face refinement

### MiniMax H3 quality-protection layers

Both GUIs keep this important section expanded for H3 and expose three independent, combinable layers:

- **Dynamic Sigma (recommended)** is the established default. It uses the cached empty-prompt state and one no-gradient DiT pass on each scheduled step. Its cadence is configurable: `1` means every step, `4` means 25% of steps, and `10` means 10%. This is the mechanism with successful long-run quality evidence; enabling it requires rebuilding only the Caption/Text Cache once.
- **Ostris Assistant (alpha)** loads [`ostris/minimax_h3_training_adapter`](https://huggingface.co/ostris/minimax_h3_training_adapter) as a frozen live helper. The roughly 155 MB helper is downloaded on first use, reused from the Hugging Face cache, disabled during previews, and never merged into the user's saved LoRA. It does not itself require an empty-prompt cache.
- **Drift/Base Preservation (optional alpha)** temporarily bypasses the user's trainable LoRA on its scheduled steps and adds a small MSE penalty against either the base-plus-assistant reference or the base alone. Start with strength `0.05` every `10` steps. This is a separate third mechanism, not another name for Dynamic Sigma.

The editable presets are **Proven Quality** (Dynamic Sigma every step), **Experimental Balanced** (assistant plus Dynamic Sigma every 10 steps), **Experimental Strong** (assistant plus Dynamic Sigma every 4 steps), and **Maximum Protection** (assistant, Dynamic Sigma every step, and the drift check). Enabling both scheduled protections can add two no-gradient predictions when their cadences coincide, for three total DiT forwards on that step.

The helper consumes Ostris's published adapter and naming format directly; Dynamic Sigma remains this project's independent cached-text Musubi adaptation. The assistant and combined configurations have CPU regression coverage but still need real ConvRot GPU measurements for speed, VRAM, and long-run output quality.

- **H3 Dynamic Sigma protection:** enabled by default for new H3 recipes. H3 is guidance-distilled even though it is not a smaller distilled model: normal inference already contains the learned guidance behavior and therefore uses guidance 1.0. Longer LoRA training can weaken this behavior. The trainer first performs an empty-prompt prediction without gradients, then trains the captioned prediction toward `unconditional + effective_scale × (normal target - unconditional)`. The recommended Sigma schedule uses `effective_scale = 1 + (4 - 1) × sigma`: protection is strongest at noisy timesteps and fades toward ordinary training near the clean result, avoiding amplification of unpredictable low-noise residuals. This follows Ostris's updated [sigma-balanced contrastive guidance](https://github.com/ostris/ai-toolkit/commit/1e1418b22c) in an independent cached-text Musubi adaptation. It adds one DiT forward per step but no second backward graph. The separately published helper and hybrid choices are described above.
- **DOP:** cached class-caption preservation is available with the same trigger/class validation and staged overrides as Krea 2. It performs three DiT predictions per batch and is substantially slower.
- **Adapter weight noise:** applies only to trained LoRA weights after optimizer steps. Start with a gentle value and compare fixed prompts.
- **Depth anchor:** reconstructs H3 clean latents using the native `clean - noise` prediction, differentiably decodes an image, and compares frozen Depth Anything features. Its dedicated GUI section can place the ~5 GB video VAE on an automatically selected secondary CUDA GPU while the DiT and Depth Anything remain on the training GPU. Cross-device tensor copies preserve the gradient back to the LoRA. Keeping the VAE resident avoids moving its weights repeatedly, and **Run Depth Every N Steps** can trade less frequent structural correction for lower average cost. The split-device path fails instead of silently falling back when PyTorch sees only one GPU. It is implemented but still requires a real dual-GPU validation; keep the same-GPU path as the compatibility baseline until then.
- **Face refinement:** an experimental DRaFT-K stage can refine an existing H3 LoRA using analyzed face references. It defaults to 512px, `draft_k=1`, and 35 swapped blocks. The face-model selector accepts both the GUI-downloaded `recognition/model.onnx` + `detection/model.onnx` layout and standard InsightFace AntelopeV2 folders containing `glintr100.onnx` + `scrfd_10g_bnkps.onnx`. The Krea Turbo fixed-evaluation renderer is not applicable to H3, so inspect saved H3 checkpoints with standalone previews instead.

Quality protection has CPU target/cache/backend regression coverage but still needs a real 24 GB speed, VRAM, and long-run quality comparison. DOP, weight noise, depth anchoring, and face refinement remain disabled by default. Focused RTX 4090 smokes completed for DOP, weight-noise scheduling, depth anchoring, and the H3 DRaFT differentiable sampler/VAE/real-AntelopeV2-reward/optimizer/save path. The intentionally crude 256px, two-denoising-step face smoke did not produce a detector-quality generated face, but the real reward's missed-detection fallback remained differentiable and completed the update with a finite gradient. Useful full-resolution face-refinement recipes and long runs still need validation. Downloading the separately licensed face models remains an explicit user action when a compatible existing InsightFace folder is not selected.

### Future work: native five-frame differentiable face refinement

The current H3 face-refinement update intentionally uses one temporal token. This keeps the DRaFT backward pass small enough to experiment with on a 24 GB card. The optional **Native five-frame + center frame** setting in the GUI is an inference-only quality preview: it runs an additional no-gradient five-frame decode when a preview is saved, but it does not change the training gradient.

A future training mode could score several native frames in one refinement update and aggregate the identity reward before backpropagation. That would better match the model's native temporal representation, but it would also increase denoising time, VAE memory, and activation pressure. It should remain opt-in, warn clearly about VRAM and speed, preserve the one-frame fallback, and record the frame count and reward aggregation policy in checkpoint metadata. Until that work is validated, use five-frame mode only to inspect preview quality and keep differentiable refinement on one frame.

Tracking issue: [experimental native five-frame differentiable MiniMax H3 face refinement](https://github.com/diodiogod/musubi-tuner_simple_GUI/issues/1). Keep the current one-frame path as the default until that experiment has measured VRAM, speed, and identity-reward stability.

## Deliberate limitations

- still images only; no video clips, reference media, or audio training
- LoRA only; no LoHa, LoKr, or full-model training
- batch size 1
- no `torch.compile`, FP8-base conversion, or standard device-to-host block swap
- only the published pruned ConvRot INT8 FL2VA tensor contract is accepted

These restrictions keep the first version small, auditable, and suitable for later replacement by the final upstream interfaces. The loader validates every quantized projection, scale, marker, dtype, and model key before training instead of silently accepting a similar checkpoint.

## Why this is separate from upstream R1

[musubi-tuner PR #1018](https://github.com/kohya-ss/musubi-tuner/pull/1018) is a broader BF16 implementation with video/audio topology, generation, scheduled sampling, and an optional video-only supervision policy. As of its 2026-08-04 head it still explicitly defers ConvRot, prequantized INT8, pruned AdaLN, and NVFP4/AWQ support. This fork reuses the compatible H3 architecture and cache contracts while isolating the experimental INT8 image path so it can later be replaced or reconciled without disturbing Wan, Flux.2, or Krea 2.

The still-image flow and compact-text-encoder behavior were also compared with [Fizgig v3.2.0](https://github.com/shootthesound/Fizgig/releases/tag/v3.2.0). Fizgig deliberately ships image-only, batch-1 H3 LoRA training without previews and reports its own 24 GB support as still being tested. This fork differs by loading the published pruned ConvRot INT8 transformer directly rather than downloading the BF16 transformer and converting it for training.

## First-run checklist

1. Close ComfyUI and other CUDA applications so the training GPU is empty.
2. Cache one small image and one short caption.
3. Set one training step, rank 4 or 8, batch size 1, `30` swapped blocks, and BF16 ConvRot backward.
4. Confirm the pruned checkpoint passes strict inspection without allocating a BF16 copy.
5. Confirm one optimizer step finishes, the loss is finite, the saved LoRA tensors are nonzero, and peak VRAM stays below the card limit.
6. Only then try rank 16 and a normal dataset. Test previews, DOP, depth, and face refinement separately so a failure has one clear cause.

If the one-step smoke fails in the loader or ConvRot backward on another setup, report the exact checkpoint, GPU, and traceback rather than weakening the checkpoint checks or falling back to a full BF16 download.
