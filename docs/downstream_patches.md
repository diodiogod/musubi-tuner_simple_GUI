# Downstream Musubi patches

This GUI repository vendors Musubi source and periodically imports upstream
snapshots as squashed commits. A normal Git merge from `kohya-ss/musubi-tuner`
is therefore unsafe: the shared merge base predates the GUI and upstream does
not contain the GUI files.

## Current upstream baseline

- Upstream repository: `https://github.com/kohya-ss/musubi-tuner`
- Audited baseline: `v0.3.4` / `30c658c` (2026-06-24)
- Audit date: 2026-07-11
- Shared trainer and parser matched the baseline at audit time.
- Krea 2 files intentionally differ because this fork adds pre-quantized FP8,
  projector-patch, and Turbo-preview support.

## Sync procedure

1. Fetch `upstream` and run `python tools/audit_upstream.py`. The command is
   read-only and uses `UPSTREAM_BASELINE.json` to flag protected-path overlap.
2. Identify the newest release tag.
3. Compare vendored areas directly with `git diff HEAD upstream/<ref> -- src/musubi_tuner`.
4. Classify every difference as upstream change, existing downstream patch, or
   unrelated model-family change. Do not merge upstream wholesale.
5. Import upstream changes in a dedicated commit, preserving the integrations
   listed below.
6. Run the focused tests plus CLI `--help` smoke tests before feature work.
7. Update `UPSTREAM_BASELINE.json` and the baseline below.

## Maintained integrations

### Krea 2 FP8, projector patch, and Turbo previews

- Primary files: `krea2_encoder.py`, `krea2_utils.py`,
  `krea2_train_network.py`, and `krea2_generate_image.py`.
- These predate the perceptual-training work and must not be overwritten by an
  upstream snapshot.

### Adapter weight noise

- Implementation: `training/weight_noise.py`.
- Upstream seam: Krea overrides the existing
  `NetworkTrainer.on_post_optimizer_step` and `extra_step_logs` hooks; the
  shared upstream trainer remains byte-for-byte unchanged.
- Krea CLI definitions: `krea2_train_network.py`.
- GUI command wiring: `backends/krea2.py`.
- Disabled invariant: `--weight_noise_sigma 0` performs no mutation and does
  not instantiate the injector.
- Tests: `tests/test_weight_noise.py`.

### Krea 2 perceptual depth anchor

- Implementation: `perceptual/depth_anchor.py`.
- Krea integration seam: the model trainer's existing `process_batch` override;
  the shared upstream loss implementation remains unchanged.
- Uses the Krea rectified-flow identity `x0 = xt - t * velocity`, then the
  frozen Qwen-Image VAE and Depth Anything V2.
- Ground-truth depth targets are cached in a bounded CPU LRU keyed by clean
  latent content. Predicted depth remains differentiable.
- Disabled invariant: `--depth_anchor_weight 0` takes the exact upstream
  `process_batch` path and does not load, download, or instantiate a perceptor.
- Tests: `tests/test_depth_anchor.py` and
  `tests/test_krea2_backend_regularization.py`.
- Effective regularization settings are embedded in saved LoRA metadata under
  `ss_krea2_*` keys for reproducibility and job-history audits.
