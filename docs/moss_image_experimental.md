# MOSS-Image training (experimental)

This integration adapts [Kohya's sd-scripts MOSS implementation](https://github.com/kohya-ss/sd-scripts)
to this project's Musubi training loop and both interfaces. It is for ordinary
text-to-image LoRA training, not MOSS image-edit or control-adapter training.
It is experimental: long-run quality and recommended settings still need testing.

## What has been tested

A short smoke test on an RTX 4090 used the real ConvRot INT8 Base checkpoint,
W4A8 Qwen3.5 encoder and small-decoder Flux.2 VAE. At 256×256 with 30 blocks
swapped, two LoRA updates completed with DOP and depth enabled, including a
scheduled sample between updates. A standalone preview loaded the saved LoRA.
The saved training state reloaded and completed another update. A separate
1024×1024 LoRA update completed with rank 32 and AdamW 8-bit, with DOP and
depth disabled. One LoKr update and its standalone preview also completed.
One 512×512 face-refinement update detected a face and produced a nonzero
gradient. These checks establish that the paths run; they do not measure LoRA
quality, guarantee memory use in other configurations, or validate every
checkpoint format, LoHa, pose-aware refinement, or secondary-GPU depth.

REPA was also checked on the real ConvRot Base checkpoint with DINOv2 Small
at teacher size 518: two 256px rank-16 updates with 30 swapped blocks completed,
then a saved-state resume completed another update. TensorBoard recorded the
separate losses. One additional update combined REPA, DOP and depth. CPU tests
check REPA-only gradients into the adapter/head, frozen teachers, rectangular
buckets, checkpointing, bounded cache reuse and projection-state restoration.
These are wiring and backward-pass checks, not a trained-quality comparison.

## Files

- **DiT:** MOSS-Image Base BF16 or the ComfyUI `moss_image_base_int8_convrot.safetensors`.
  The prequantized ConvRot file is detected automatically; do not turn on
  load-time quantization again for it. The SFT checkpoint can be selected too,
  but Base is the preferred starting point for training.
- **Text encoder:** Qwen3.5-9B. BF16, ComfyUI ConvRot INT8, and ComfyUI
  `asym_w4a8_int8` W4A8 checkpoints are supported. W4A8 uses a portable decoder
  that is experimental and can be slower than ComfyUI's fused implementation.
  Other GPTQ or quantization formats are not automatically supported.
- **VAE:** Flux.2 VAE, including standard and small-decoder diffusers/ComfyUI
  variants. It is unrelated to the MiniMax H3 and Krea 2 VAEs.
- **Tokenizer:** leave the optional tokenizer location blank to fetch the small
  tokenizer from the official MOSS repository, or point it to a local tokenizer
  directory. Model weights are not automatically downloaded by this step.

## Starting a run

1. Select **MOSS-Image (Experimental)**, choose the three model files, and
   create an image dataset with captions. Video datasets and edit conditioning
   are not supported by this trainer.
2. Rebuild the image/latent and caption/text caches the first time. Cache files
   are separate from other architectures. Changing captions or the text encoder
   requires refreshing the text cache; changing images, resolution, or VAE
   requires refreshing the latent cache.
3. Start with BF16 compute, gradient checkpointing, SDPA, and a conservative
   number of swapped blocks. More swapping reduces graphics-memory use and
   typically slows training. The initial 30-block UI value is experimental,
   not a verified memory guarantee. MOSS has 42 main blocks and supports up to
   40 swapped blocks. Text-encoder CPU streaming is separate from DiT block
   swapping and can reduce caption-caching VRAM at the cost of speed.
4. The recommended MOSS pretraining noise schedule uses its published defaults:
   logit mean −1.09, standard deviation 0.95, cutoff 0.95, and a 1% uniform
   mixture. Disable the MOSS schedule to use the general timestep controls.
5. Configure sample cards as images. Standard MOSS preview defaults are 50
   denoising steps, guidance 3, and shift 5. Block 22 is skipped in the
   negative guidance pass by default; this does **not** skip it during
   training. Scheduled previews run between training steps, while standalone
   previews can optionally apply the selected LoRA.

Starting from LoRA weights creates a fresh optimizer. Resuming a compatible
saved state restores the optimizer and run state through Musubi's normal
resume mechanism.

## Optional experiments

- **DOP** caches an additional class caption and adds a frozen-base comparison
  pass. Every caption must contain the configured trigger. It uses more compute
  and should not be assumed to prevent all overtraining.
- **Depth anchor** compares estimated structure in the decoded prediction with
  the training image. It requires a differentiable VAE decode and a frozen Depth
  Anything model, so it can add substantial memory and time. The VAE can run on
  the training GPU, a secondary GPU, or CPU; the depth model stays on the
  training GPU. These choices are experimental and have not been benchmarked
  for MOSS.
- **Adapter weight noise** changes the LoRA after optimizer updates; leave it
  disabled unless deliberately testing it.
- **REPA visual guidance** adds a frozen DINOv2 image-understanding teacher to
  ordinary image training. It aligns image tokens from one early MOSS layer
  with the teacher's patch features through a learned projection head. This is
  an independent Musubi implementation informed by the
  [REPA paper](https://arxiv.org/abs/2410.06940) and
  [SimpleTuner's image REPA approach](https://github.com/bghira/SimpleTuner/blob/9e8ff82ee2b24d44bcb692f4abad01d3d9610549/documentation/experimental/IMAGE_REPA.md).
  It is disabled by default and available separately from DOP and depth in both
  interfaces. In Modern UI, open **Regularization → MOSS · REPA visual learning
  guidance** and click **?** beside the switch for a practical explanation.
  In Classic GUI, use the **MOSS · REPA visual guidance** panel and expand
  **Show REPA settings**. It works with BF16 and ConvRot INT8 MOSS training,
  including gradient checkpointing and fixed block swapping, on one training
  GPU/process. FP16 mixed precision is not supported by this experiment.
- **Face refinement** is an optional DRaFT-K stage using reference faces and an
  existing LoRA. It optimizes generated faces toward an identity reward; it is
  not a replacement for training on detailed face crops. A single MOSS update
  has been smoke-tested; its long-run behavior and pose-aware option are not
  validated yet.

Try these techniques separately first so you can tell which change helped.
Save an earlier checkpoint before continuing a long run.

### REPA settings and memory

The experimental starting settings are strength **0.5**, layer **8** (zero-based),
teacher image size **518** and **facebook/dinov2-small**. SimpleTuner's MOSS
example uses DINOv2 **Giant**; our Small default trades teacher capacity for
lower memory use and is not a measured quality recommendation. Other standard
Hugging Face DINOv2 sizes and local model folders are supported. Other encoder
families and DINO register-token variants are rejected. The teacher downloads
on first use; no additional dependencies or dataset annotations are required.

REPA uses the same clean cached latents that supply the normal training target.
On a cache miss, the frozen Flux.2 VAE decodes one image **without gradients**,
then DINO reads the resized image. The VAE and DINO run sequentially and return
to CPU before the training forward. This avoids depth guidance's differentiable
VAE graph, but high-resolution decoding and the teacher still need memory.
Select **cpu** for teacher compute to run both helpers on CPU when GPU memory
is tight; this can be very slow. Teacher size **224** is a cheaper experiment
than 518 and does not change your dataset resolution.

A bounded CPU RAM cache remembers teacher features for **64** images by default;
0 disables reuse. Features are keyed by exact latent contents and spatial shape.
They are rebuilt on demand after restarting, so **neither text nor latent
recaching is needed when enabling REPA**. Normal image/VAE/resolution cache
rules still apply when those inputs change. Rectangular image buckets use
two-dimensional spatial alignment; caption tokens never enter the REPA loss.

The projection head participates in optimizer updates and full Accelerate state
saves. Resume with the same REPA configuration to restore it and its optimizer
state. To enable/disable REPA or change its encoder, layer or image size, start
from LoRA weights with a fresh optimizer. Exported LoRAs contain only the normal
adapter weights; inference requires no DINO or projection head.

Track `loss/diffusion`, `loss/repa`, `loss/repa_weighted` and
`repa/cosine_similarity` in TensorBoard/W&B. The logged alignment loss is
`1 - cosine similarity`, a constant shift of the negative cosine objective with
identical gradients. Its numerical scale differs from plain diffusion loss.
REPA can help learning, but does not guarantee protection against overtraining,
forgetting or reduced generation quality. Long-run MOSS quality is unverified.
