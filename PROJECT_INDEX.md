# Musubi Tuner Simple GUI - Project Index

*Comprehensive file index for the Windows-focused multi-mode GUI wrapper around musubi-tuner*

## Architecture Overview

**Two-layer architecture** - the repo is split between a local desktop GUI layer and the upstream musubi-tuner training/runtime layer:

1. **Desktop GUI** (`musubi_tuner_gui.py`) - Tkinter application, mode-aware forms, monitor, samples, job history, conversion tools
2. **GUI Backend Adapters** (`backends/`) - translate GUI settings into cache/train command lines per supported mode
3. **Root Wrapper Scripts** (`*_train_network.py`, `*_cache_*.py`, `*_generate_*.py`) - thin entrypoints that call the packaged `src/` modules
4. **Packaged Training Core** (`src/musubi_tuner/training/`) - shared parser, trainer, logging, checkpointing, sampling hooks
5. **Model-Family Implementations** (`src/musubi_tuner/<family>/`, `src/musubi_tuner/*_train*.py`) - architecture-specific training, caching, and inference flows
6. **Shared Utilities** (`src/musubi_tuner/utils/`, `dataset/`, `modules/`, `networks/`) - dataset loading, LoRA handling, save logic, model utilities
7. **Docs + Assets** (`docs/`) - upstream musubi-tuner docs plus GUI screenshots and user-facing references

**Key architectural rules:**

- GUI behavior should prefer living in `musubi_tuner_gui.py`, GUI helper modules such as `dataset_config_builder.py` / `prompt_library.py`, and `backends/` rather than patching upstream-style `src/` training logic unless the backend itself genuinely needs a behavior change
- `backends/` is the command-construction layer; it decides which wrappers/scripts run for each mode
- GUI-generated Accelerate training commands explicitly use one process/GPU; this prevents machines with an additional or integrated GPU from silently entering unsupported Windows distributed mode
- Root-level wrapper scripts mirror packaged modules so users can launch workflows from the repo root without remembering `src/` module paths
- The GUI keeps musubi-tuner’s underlying CLI workflow intact and mainly adds state management, UX, monitoring, and automation around it
- Mode-specific required fields and visible sections are driven from the selected training mode inside the GUI, not from separate apps
- Staged training, sample prompts, local job history, and post-run convenience behaviors are GUI-side orchestration features

## Supported GUI Modes

4 primary GUI modes currently ship in the app:

| Mode | GUI Backend | Main Training Wrapper | Primary Docs |
|------|-------------|-----------------------|--------------|
| Wan 2.2 | `backends/wan.py` | `wan_train_network.py` | `docs/wan.md` |
| Flux.2 Klein | `backends/flux2.py` | `flux_2_train_network.py` | `docs/flux_2.md` |
| Flux.2 Dev | `backends/flux2.py` | `flux_2_train_network.py` | `docs/flux_2.md` |
| Krea 2 | `backends/krea2.py` | `krea2_train_network.py` | `docs/krea2.md` |

**Mode-specific GUI behavior includes:**

- Wan 2.2: dual low/high-noise workflows, combined vs separate runs, timestep-boundary handling, I2V/T2V controls
- Flux.2: single-model DiT workflow, Flux model-version selection, Qwen3/Mistral text encoder selection, and optional DOP class preservation for Klein 4B/9B
- Krea 2: RAW DiT flow, optional Turbo DiT sampling path, projector patch handling, Krea-specific timestep defaults, experimental small-dataset generalization controls (adapter weight noise and automatic depth anchoring), and optional DOP class preservation

## Documentation Files

**README.md** - Main GUI docs, installation, supported modes, workflow overview
**CLAUDE.md** - Repo-specific dev instructions inherited from the user’s workflow
**PROJECT_INDEX.md** - This project map

### User / Reference Docs (`docs/`)

- `wan.md` - Wan training and inference reference
- `flux_2.md` - Flux.2 reference
- `krea2.md` - Krea 2 reference
- `dataset_config.md` - Dataset config reference
- `sampling_during_training.md` - Sampling behavior reference
- `advanced_config.md` - Advanced training flags and behaviors
- `loha_lokr.md` - LoHa / LoKr notes
- `dop.md` - Krea 2 / FLUX.2 Klein Differential Output Preservation setup, monitoring, and staged behavior
- `musubi-tuner-gui.png` - Main GUI screenshot

### Other Model Docs in `docs/`

The repo also carries upstream musubi-tuner docs for additional architectures and utilities not all exposed directly in this GUI, including:

- `framepack.md`, `framepack_1f.md`
- `hunyuan_video.md`, `hunyuan_video_1_5.md`
- `ideogram4.md`
- `kandinsky5.md`
- `qwen_image.md`
- `zimage.md`
- `tools.md`, `torch_compile.md`, `block_swap.md`

## Core Files

**`musubi_tuner_gui.py`** - Main Tkinter application; the center of GUI behavior
**`dataset_config_builder.py`** - Visual/raw dataset TOML builder with validation and safe round-tripping
**`prompt_library.py`** - Global prompt-library persistence, searchable gallery, job migration, and test thumbnails
**`UPSTREAM_BASELINE.json`** - Machine-readable upstream Musubi baseline and protected downstream paths
**`README.md`** - Main project docs
**`LAUNCH_GUI.bat`** - Windows launcher
**`pyproject.toml`** - Package metadata and Python dependencies
**`Base_SETTINGS.json`** - Example/default persisted settings snapshot

## GUI Implementation

### Main Desktop App

- `musubi_tuner_gui.py` - everything GUI-side: tab layout, validation, process launch, output parsing, monitoring, sample gallery, job history, staged training, conversion, setup helpers
- Recent Jobs separates ordinary continuation from failed-run recovery: recovery validates complete Accelerate model/optimizer/scheduler/RNG state plus a numbered epoch/step position, keeps the original output identity, and carries bounded loss history into the resumed monitor
- Staged artifact names are derived internally from a stable base output name; stage execution does not replace the main Output Name field, preventing repeated labels across retries
- The shared trainer reconstructs `global_step`, starts an epoch checkpoint at the following epoch, and skips already-consumed batches for a step checkpoint; unnumbered end-state folders are not advertised as exact positional recovery

### Resume Semantics — Preserve This Invariant

- **Continuation is additive:** the existing **Load as Continuation / Resume** workflow branches to a new output name and trains the configured number of epochs as additional work. It must never pass `--resume_exact_position`.
- **Failed-run recovery is positional:** only the verified **Recover Failed Run (True Resume)** workflow keeps the original run identity and passes `--resume_exact_position`; an epoch-4 state must continue at epoch 5 toward the original total (for example 5/16), not start another 1/16 block.
- Do not infer these two workflows from `--resume` alone. Preserve the explicit opt-in flag, state-folder validation, and tests when importing upstream trainer/parser changes.
- `dataset_config_builder.py` - standalone dataset-config editor used from the Models page
- `prompt_library.py` - user-level prompt gallery stored outside the repo; prompts remain copied into run snapshots for reproducibility

### GUI Backend Adapters (`backends/`)

- `_common.py` - shared command-building helpers and argument assembly
- `wan.py` - Wan command construction
- `flux2.py` - Flux.2 command construction
- `krea2.py` - Krea 2 command construction

Krea 2 also forwards optional weight-noise and depth-anchor controls. Both are explicit no-ops at their default strength of `0`.
Krea 2 and FLUX.2 Klein also forward disabled-by-default DOP settings to both the text-cache and training commands.

**Backend responsibility split:**

- GUI gathers/validates settings
- backend modules convert those settings into concrete cache/train/generate commands
- root wrapper scripts execute packaged `src/` modules

## Root Wrapper Scripts

These exist at repo root mainly as user-facing launch points that import the packaged implementations under `src/musubi_tuner/`.

### Training Wrappers

- `wan_train_network.py`
- `flux_2_train_network.py`
- `krea2_train_network.py`
- `hv_train_network.py`
- `hv_1_5_train_network.py`
- `qwen_image_train_network.py`
- `zimage_train_network.py`
- `fpack_train_network.py`
- `kandinsky5_train_network.py`

### Cache Wrappers

- `cache_latents.py`, `cache_text_encoder_outputs.py`
- `wan_cache_latents.py`, `wan_cache_text_encoder_outputs.py`
- `flux_2_cache_latents.py`, `flux_2_cache_text_encoder_outputs.py`
- `krea2_cache_latents.py`, `krea2_cache_text_encoder_outputs.py`
- additional family-specific cache wrappers for Hunyuan, Qwen Image, Z-Image, Kandinsky, FramePack, etc.

### Generation / Utility Wrappers

- `wan_generate_video.py`
- `flux_2_generate_image.py`
- `krea2_generate_image.py`
- `qwen_image_generate_image.py`
- `zimage_generate_image.py`
- `convert_lora.py`
- `merge_lora.py`
- `qwen_extract_lora.py`
- `lora_post_hoc_ema.py`

## Packaged Training Core (`src/musubi_tuner/`)

### Shared Training System

- `training/trainer_base.py` - central shared training loop and checkpoint/sample orchestration
- `training/parser_common.py` - shared CLI argument definitions
- `training/accelerator_setup.py` - Accelerate setup helpers
- `training/sampling_prompts.py` - prompt-driven sample support
- `training/timesteps.py` - timestep logic
- `training/weight_noise.py` - isolated adapter-only Gaussian weight-noise implementation used by the Krea trainer hook
- `training/dop.py` - shared caption replacement/signature validation, temporary adapter bypass, and preservation-loss implementation used by Krea 2 and FLUX.2 Klein

### Experimental Differential Output Preservation

- Krea 2 and FLUX.2 Klein text-cache scripts store a second class-caption embedding plus a trigger/class signature; stale or mismatched caches fail clearly. DOP uses a separate post-primary backward pass whose gradients join the same optimizer step, avoiding simultaneous primary/depth/DOP graphs
- Model trainers reuse the normal noisy latent/timestep for a frozen-base class prediction and a LoRA-enabled class prediction, logging `loss/dop` and `loss/dop_weighted`
- GUI validation requires a distinct trigger and class, explains the speed cost, enables safe first-run recaching, and shows DOP participation in Live Monitoring
- Standard staged-training rows provide inherit/enable/disable plus optional strength/trigger/class overrides; DOP stages validate and reuse matching dual-caption caches and encode only stale/missing entries, with an explicit full-recache option
- Technique reference: `ostris/ai-toolkit`; implementation is an independent cached-text Musubi adaptation

### Experimental Perceptual Training

- `perceptual/depth_anchor.py` - differentiable Depth Anything V2 anchor, rectified-flow clean-latent reconstruction, bounded target-depth cache, and optional keep-on-GPU helper residency for faster lower-resolution stages
- Krea integration stays in `krea2_train_network.py` through existing `process_batch`, `on_post_optimizer_step`, `extra_step_logs`, and `extra_metadata` overrides; the shared upstream trainer remains unchanged
- Ground-truth depth targets are generated automatically from cached dataset latents during training and cached in CPU RAM; users do not prepare depth maps. Oversized latents are reduced only for the depth VAE decode to avoid wasting VRAM above the perceptor's configured resolution
- The live monitor distinguishes the main `steps:` bar from model-loading bars, graphs combined loss, and shows depth loss/contribution; Krea sampling offloads the depth helper, caches only Turbo (RAW restores from disk), falls back to full streaming when RAM headroom is unsafe, and uses tiled VAE decoding only as an OOM fallback
- Saved LoRAs record effective settings in `ss_krea2_*` metadata

### Experimental Face Refinement

- `face_refinement/` - AntelopeV2 per-image reference preflight/outlier scoring, Krea LoRA/trigger validation, differentiable face reward, model download helper, and truncated DRaFT sampling
- `face_refinement/pose.py` - optional landmark-only yaw/pitch/roll estimates, broad virtual pose buckets, confidence, and prompt-tag parsing; no extra model weights
- `face_refinement/pose_plan.py` - pose-goal presets, offline prompt suggestions, share normalization, sparse-bucket safeguards, and per-pose target/plateau tracking
- `krea2_face_refinement.py` - standalone Krea face-refinement trainer that consumes/emits complete LoRA files
- `krea2_face_evaluate.py` + `face_refinement/evaluation.py` - read-only fixed-suite Turbo scoring for identity, matching pose, requested-pose success, detection, and baseline deltas
- `backends/krea2_face.py` - command construction for typed face-refinement stages
- `backends/krea2_face_eval.py` - fixed prompt/seed Turbo suite preparation plus generation/scoring command construction
- The main GUI has a dedicated Face Refinement workspace with persistent setup/status cards, an embedded latest-evaluation report, per-pose image galleries, configurable intermediate saves, weak-pose plan actions, and staged-run shortcuts
- Turbo evaluation batch-encodes unique prompts before loading the DiT, reuses embeddings across seeds, and correctly dequantizes pre-scaled FP8 weights before merging a LoRA
- Face settings and prompts are persisted in GUI/job JSON snapshots; no second dataset TOML is exposed
- Staged handoff remains type-aware: standard→standard uses state directories, standard↔face uses LoRA files, and a first-stage face run uses an explicitly validated existing LoRA

### Shared Utilities

- `utils/train_utils.py` - checkpoint/state naming, save/remove policies
- `utils/huggingface_utils.py` - HF uploads/download helpers
- `utils/model_utils.py` - dtype and model-save helpers
- `utils/lora_utils.py` - LoRA merge/load utilities
- `utils/safetensors_utils.py` - safetensors helpers, split-file loading
- `utils/sai_model_spec.py` - metadata embedding/spec helpers
- `utils/device_utils.py` - device synchronization helpers
- `utils/image_utils.py` - shared image helpers

### Dataset System

- `dataset/config_utils.py` - dataset config parsing/validation
- `dataset/image_video_dataset.py` - core dataset loader
- `dataset/cache_io.py` - cache read/write logic
- `dataset/bucket.py` - bucket handling
- `dataset/media_utils.py` - media helpers
- `dataset/datasources.py` - dataset source abstraction
- `dataset/architectures.py` - architecture tagging/selection

### Network / LoRA System

- `networks/lora.py` - base LoRA implementation
- `networks/loha.py`, `networks/lokr.py` - alternate network types
- `networks/lora_wan.py`, `lora_flux_2.py`, `lora_krea2.py`, `lora_qwen_image.py`, `lora_zimage.py`, etc. - family-specific network glue
- `networks/convert_*_to_comfy.py` - conversion helpers

## Model Families in `src/musubi_tuner/`

Each family generally follows the same pattern:

1. root/module-level cache wrapper(s)
2. training wrapper(s)
3. generation wrapper(s)
4. a dedicated package or helpers for model-specific runtime details

### Major families present in this repo

- `wan/` - Wan modules, configs, tokenizers, model loading
- `flux/`, `flux_2/` - Flux family implementations
- `krea2/` - Krea 2 encoder, MMDiT, sampling, utilities
- `qwen_image/` - Qwen Image model code
- `zimage/` - Z-Image model code
- `kandinsky5/` - Kandinsky 5 implementation
- `ideogram4/` - Ideogram 4 implementation
- `hidream_o1/` - HiDream-O1 implementation
- `hunyuan_model/`, `hunyuan_video_1_5/` - Hunyuan video model code
- `frame_pack/` - FramePack workflows/utilities

## Additional GUI Package (`src/musubi_tuner/gui/`)

This is a separate packaged GUI implementation path from upstream musubi-tuner and is distinct from this repo’s main desktop file:

- `gui.py` - packaged GUI implementation
- `config_manager.py` - GUI config persistence
- `i18n_data.py` - localization data
- `gui.md`, `gui.ja.md` - GUI docs
- `gui_implementation_plan.md` - implementation notes/planning

For this fork, `musubi_tuner_gui.py` is the practical entrypoint users work with.

## Utility Systems

### Conversion / Merge Tools

- `convert_lora.py` / `src/musubi_tuner/convert_lora.py`
- `merge_lora.py` / `src/musubi_tuner/merge_lora.py`
- `qwen_extract_lora.py`
- `lora_post_hoc_ema.py`
- `tools/merge_krea2_filterbypass.py`

### Sample / Caption Helpers

- `caption_images_by_qwen_vl.py`
- sample generation wrappers for each model family

### Upstream Sync Tracking

- `UPSTREAM_BASELINE.json` records the audited Musubi release/commit and downstream-protected paths
- `tools/audit_upstream.py` provides a read-only report of changes since that baseline and flags overlap with protected Krea/perceptual files
- `docs/downstream_patches.md` documents the selective-sync procedure and maintained integration seams; upstream must not be merged wholesale because this fork imports squashed snapshots

### Focused Regression Tests

- `tests/test_weight_noise.py` - noise behavior, frozen-parameter protection, and norm bounding
- `tests/test_depth_anchor.py` - clean-latent reconstruction, gradient direction, and target-depth caching
- `tests/test_krea2_backend_regularization.py` - GUI/backend command forwarding and disabled defaults
- Upstream gradient diagnostics (`--log_grad_metrics`) are available from the Logging panel and remain disabled by default; the audited upstream baseline is `8934cfb` (2026-07-22)

## Typical GUI Workflow

1. User selects a training mode in `musubi_tuner_gui.py`
2. GUI validates mode-specific required fields and settings
3. A backend adapter in `backends/` converts settings into command sequences
4. Root wrapper scripts launch packaged `src/musubi_tuner/...` modules
5. GUI monitors stdout, tracks loss/epochs/steps/VRAM, and watches sample outputs
6. Job history, staged training, continuation behavior, and optional local post-run renames remain GUI-side

## Current Repo Focus

This fork is focused on:

- making musubi-tuner easier to use from a single Windows desktop app
- supporting several model families from one interface
- improving staged training, sample management, continuation workflows, monitoring, and local productivity features without drifting too far from upstream backend behavior
- experimentally improving Krea 2 LoRA generalization on small datasets while keeping the implementation isolated and future upstream syncs auditable
