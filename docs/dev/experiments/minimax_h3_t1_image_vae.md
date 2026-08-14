# Developer Experiment: MiniMax H3 Direct Single-Frame VAE

**Status:** concluded; optional experiment, not the default implementation  
**Test date:** 2026-08-14  
**Hardware:** NVIDIA RTX 4090 (24 GB)

## Question

The community released an image-specialized MiniMax H3 VAE that decodes a single
temporal latent (`T=1`) directly. Could it replace our temporal compatibility
decode, improve one-frame image quality, or reduce generation/training time?

References:

- [Mamad8/MiniMax-H3-Image-VAE](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE)
- [H3 as a single-image edit model](https://www.reddit.com/r/StableDiffusion/comments/1vo1ab3/h3_as_a_singleimage_edit_model/)

## Checkpoint

Downloaded beside the existing ComfyUI H3 VAEs:

`J:\stablediffusion1111s2\Data\Packages\ComfyUI6\models\vae\minimax_h3_t1_image_vae_step1597.safetensors`

- File size: 5,207,808,784 bytes
- SHA-256: `6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86`
- It is a normal merged H3 VAE checkpoint, not a new DiT or text encoder.
- It is intended for image-only `T=1` decoding; the model card warns against
  using it for multi-frame video reconstruction.

## Reproducible test

The same compact image-generation run was used for all outputs:

| Setting | Value |
| --- | --- |
| DiT | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| LoRA | `Fares_Fares-1024-MinMax-000001.safetensors` |
| Prompt | `Fares Fares, a photorealistic high-detail portrait photograph of the same man, natural skin texture, sharp eyes, detailed face, neutral expression, soft studio light, 1024x1024 still photo` |
| Resolution | 1024 × 1024 |
| Steps | 20 |
| Seed | 42 |
| LoRA multiplier | 1.0 |
| H3 block swap | 30 |
| Text-encoder block swap | 50 |

Three decode modes were compared:

1. Official video VAE with our existing `temporal_compat` mode. The `T=1`
   latent is duplicated to two temporal tokens before decoding.
2. Image-specialized VAE with direct `T=1` decoding.
3. Official video VAE with direct `T=1` decoding as a control.

Outputs were saved outside the repository at:

`J:\train\fares krea\preview_vae_compare\`

## Observations

- Official VAE + direct `T=1` showed obvious grid/temporal artifacts.
- The image-specialized VAE removed those artifacts and produced a clean image.
- On this Fares LoRA sample, the image-specialized VAE was not visibly sharper
  than the existing temporal-compatible output. The existing output retained at
  least as much apparent fine texture.
- Denoising time was approximately 34 seconds for the existing mode, 32 seconds
  for the image-specialized VAE, and 31 seconds for the official VAE direct-T1
  control. This variation is not evidence of a meaningful speed improvement;
  model/text loading dominates the full run.
- The image-specialized checkpoint is approximately the same size as the
  official video VAE, so it does not provide a meaningful memory reduction.

## Decision

Keep the existing temporal-compatible decode as the default. It already avoids
the official VAE's direct-T1 artifacts and gives comparable quality without an
additional model requirement.

The image-specialized VAE remains useful for manual ComfyUI experiments or a
future opt-in direct-T1 preview mode. It should not replace the official H3 VAE
for video generation, multimodal training, or multi-frame previews.

The experiment added an opt-in CLI switch to the compact image generator:

`--image_vae_mode {temporal_compat,single_frame}`

The default remains `temporal_compat`. No training latent recache is required
for this decoder-only comparison.
