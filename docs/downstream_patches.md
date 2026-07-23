# Downstream Musubi patches

This GUI repository vendors Musubi source and periodically imports upstream
snapshots as squashed commits. A normal Git merge from `kohya-ss/musubi-tuner`
is therefore unsafe: the shared merge base predates the GUI and upstream does
not contain the GUI files.

## Current upstream baseline

- Upstream repository: `https://github.com/kohya-ss/musubi-tuner`
- Audited baseline: upstream `main` / `8934cfb` (post-`v0.3.4`)
- Latest release at audit time: `v0.3.4` (2026-06-24)
- Audit date: 2026-07-22
- Upstream gradient diagnostics and CUDA 13.2 packaging are imported; shared trainer/parser retain only the documented exact-resume seams.
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

### Deterministic single-GPU GUI launches

- GUI backend commands in `backends/wan.py`, `backends/flux2.py`, and
  `backends/krea2.py` pass `accelerate launch --num_processes 1` explicitly.
- `musubi_tuner_gui.py` also removes inherited torchrun/MPI rank and rendezvous
  variables from subprocess environments. Even `LOCAL_RANK=0` makes Accelerate
  expect a distributed process group, which is invalid for the simple launcher.
- The current GUI monitor, VRAM controls, staged handoff, and Windows runtime are
  single-GPU workflows. Do not let Accelerate infer multi-GPU merely because a
  second or integrated adapter is visible; that can enter an unconfigured
  distributed rendezvous and fail before training starts.
- Advanced multi-GPU users can still launch the underlying Musubi CLI directly.

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
- Oversized latents are resized with their aspect ratio intact before the
  depth-only VAE decode. The decoded long side is limited to the configured
  perceptor resolution; training latents and generated samples are unchanged.
- `--keep_depth_helpers_on_gpu` optionally keeps the frozen VAE and depth
  perceptor resident across the separate DOP pass and between steps. It changes
  memory scheduling only, defaults off, and can be overridden per staged row.
- Disabled invariant: `--depth_anchor_weight 0` keeps the standard diffusion-loss
  behavior and does not load, download, or instantiate a perceptor.
- Tests: `tests/test_depth_anchor.py` and
  `tests/test_krea2_backend_regularization.py`.
- Effective regularization settings are embedded in saved LoRA metadata under
  `ss_krea2_*` keys for reproducibility and job-history audits.

### Differential Output Preservation (Krea 2 and FLUX.2 Klein)

- Shared implementation: `training/dop.py`; technique reference:
  `https://github.com/ostris/ai-toolkit`. This is an independent Musubi adaptation.
- Architecture seams: `krea2_train_network.py` and `flux_2_train_network.py`
  call the shared helper from their model-specific `process_batch` overrides.
- Cache seams: `krea2_cache_text_encoder_outputs.py`,
  `flux_2_cache_text_encoder_outputs.py`, and optional DOP tensors in
  `dataset/cache_io.py`. Normal cache keys and behavior remain unchanged when DOP is off.
- Shared-parser seam: the block marked `DOWNSTREAM` in `training/parser_common.py`
  adds disabled-by-default DOP arguments.
- Safety invariant: the class cache stores a trigger/class signature; missing,
  stale, or mismatched caches fail instead of silently training the wrong objective.
- DOP is calculated as a post-primary auxiliary backward pass. Its gradients
  accumulate before the same optimizer step, while the primary/depth graph and
  frozen helpers can be released first to control peak VRAM.
- Staged DOP caching uses `--skip_existing` with a DOP-aware validator: unchanged
  caption/signature/embedding caches are reused, while missing or stale entries are
  regenerated. The GUI retains an explicit full text re-cache control.
- `training/accelerator_setup.py` creates an init-process-group handler only
  when launcher rank variables identify a real distributed job. Merely having
  multiple physical GPUs must not put a GUI single-process run into a partially
  initialized distributed state.
- Disabled invariant: `--dop_loss_weight 0` performs no extra DiT forwards and
  does not inspect DOP cache keys.
- GUI/backend integration: `musubi_tuner_gui.py`, `backends/_common.py`,
  `backends/krea2.py`, and `backends/flux2.py`, including staged overrides.
- Tests: `tests/test_dop.py`.

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
