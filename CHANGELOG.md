# Changelog

Release notes for **Musubi Tuner Simple GUI**, including Classic GUI, Musubi
Studio, and the fork's training extensions. GUI releases follow Semantic
Versioning independently of upstream Musubi Tuner. Earlier development remains
documented in Git history; inherited Musubi package numbers are not GUI releases.

## [0.2.0] - 2026-11-24

### Added

- Train text-to-image LoRAs with **MiniMax H3 Image (Experimental)** in both GUIs. This mode uses its own checkpoint and VAE, separate caches, an optional frozen training adapter, and six-step previews. It was checked with synthetic inputs. Full-model runs are still unverified.
- Use optional `-masklabel.png` sidecars to focus the training loss in MOSS, Krea 2, FLUX.2 Klein and Dev, MiniMax H3 Image, compact MiniMax H3, and Wan 2.2 still-image training. The option is off by default and works with existing caches.
- Request a preview, save a LoRA or resumable checkpoint, or save and stop at the next completed optimizer step in either GUI.
- Choose a first and last frame for native H3 FL2VA previews, or a reference manifest for Ref2VA. The inputs work with standalone and scheduled video samples.
- Average LoRA checkpoints from the same run with post-hoc EMA in both GUIs.
- Tune the memory and speed trade-off with partial MOSS gradient checkpointing, and optionally record per-image denoising losses for dataset review in Musubi Studio.
- Choose `sigma_corrected` to apply timestep bounds and logit-mean bias in the intended direction. Existing `sigma` recipes retain their behavior.

### Changed

- Open **MOSS Heal ?** beside the model picker for INT8 and BF16 downloads and the model card.

### Fixed

- Disable LoRA dropout during training previews and restore the prior adapter mode afterward.
- Keep the Krea 2 Turbo preview LoRA in CPU memory between samples.
- Allow the Wan text encoder module to be imported without initializing CUDA.

## [0.1.2] - 2026-11-24

### Changed

- Restore the source LoRA and both strengths from the last successful export in Musubi Studio and Classic.

### Fixed

- Save to the next available numbered filename when the requested or automatically generated name is taken. Existing files are preserved, and the run log shows the filename actually saved.

## [0.1.1] - 2026-11-20

### Added

- Rescale a standard LoRA for a new ComfyUI strength in Musubi Studio or Classic. For example, save a copy of a MOSS-Image LoRA tuned at 1.8 for use at 1.0 while preserving its learned weight tensors.
- Find MOSS-Image Heal BF16 and ConvRot INT8 checkpoint downloads next to the model field in both GUIs.

## [0.1.0] - 2026-11-12

### Added

- Establish an independent GUI release version shared by Classic GUI and Musubi Studio.
- Add automated GUI version bumps, structured release notes, and README release badges.

### Changed

- Browse sampling progress by prompt and checkpoint, with thumbnail A/B selection and a pinned reference.
- Resize either comparison view, drag a locked Wipe divider, and reveal its central handle only when needed.
- Show a compact summary of the extra sample folders selected for comparison.

This first numbered GUI release includes the existing multi-model workflows,
training history and notes, staged runs, preview library, and experimental
training extensions. It does not imply a new upstream Musubi release or a change
to the bundled training algorithms.
