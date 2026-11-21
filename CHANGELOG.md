# Changelog

Release notes for **Musubi Tuner Simple GUI**, including Classic GUI, Musubi
Studio, and the fork's training extensions. GUI releases follow Semantic
Versioning independently of upstream Musubi Tuner. Earlier development remains
documented in Git history; inherited Musubi package numbers are not GUI releases.

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
