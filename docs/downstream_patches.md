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
- Technique inspiration and practical reference:
  `https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual`.
  The Musubi/Krea code is an independent adaptation, not a direct source port.
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

### Krea 2 DRaFT face refinement

- Isolated implementation: `face_refinement/` and `krea2_face_refinement.py`.
- Turbo evaluation stays isolated in `face_refinement/evaluation.py` and `krea2_face_evaluate.py`; upstream Krea generation behavior is unchanged.
- GUI/backend integration: `backends/krea2_face.py` plus typed staged-training handoff.
- Standard stages continue through Musubi state directories; face stages consume and emit full
  LoRA files and never use dataset TOML.
- The face reward is adapted from `KONAKONA666/krea-2` under Apache-2.0. Preserve
  `THIRD_PARTY_NOTICES.md`, the source header, and `licenses/Apache-2.0.txt`.
- AntelopeV2 artifacts are optional, user-initiated downloads and must never be committed or
  silently bundled.

### Exact failed-run recovery

- Isolated position parser: `training/resume_position.py`.
- Shared upstream seams: the opt-in `--resume_exact_position` definition in
  `training/parser_common.py`, plus the blocks marked `DOWNSTREAM: exact resume` in
  `training/trainer_base.py`.
- Disabled invariant: ordinary `--resume` keeps upstream/additive continuation behavior. Exact
  epoch/global-step reconstruction and dataloader skipping occur only with
  `--resume_exact_position`, which the GUI adds only for a validated **Recover Failed Run** job.
- Epoch states start at the following epoch; step states restore `global_step` and skip already
  consumed batches in the current epoch. Unnumbered state folders are never presented as exact.
- GUI integration and bounded loss-history restoration remain in `musubi_tuner_gui.py` and
  `backends/_common.py`.
- Tests: `tests/test_true_resume_position.py` and `tests/test_gui_job_history.py`.
