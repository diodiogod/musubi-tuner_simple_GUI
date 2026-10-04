# Future-idea implementation validation — 2026-10-04

Implemented the repository owner's ideas in [issue #2](https://github.com/diodiogod/musubi-tuner_simple_GUI/issues/2) (incremental cache updates) and [issue #1](https://github.com/diodiogod/musubi-tuner_simple_GUI/issues/1) (native five-frame differentiable H3 face refinement). Both Classic and Modern GUIs expose the new controls.

## Automated verification

- Full available regression suite: **852 passed**. The unrelated explicit opt-in automatic-offloading CUDA suite was excluded. Covered cache missing/matching/stale entries, corrupt and legacy files, model/precision/control/DOP/teacher changes, deliberate full rebuilds, staged text overrides, truncated native gradients, reward aggregation, metadata, and unchanged default one-frame results.
- New modules/tests passed Ruff checks; JavaScript syntax and Git whitespace checks passed.
- Classic GUI and H3 face dialog constructed successfully with hidden test windows and settings persistence isolated.
- Modern GUI displayed incremental caching enabled and one-frame training selected by default. Five-frame training and a 0.25 worst-frame weight saved and survived reload while preview quality remained one-frame. No browser errors occurred. Test settings were isolated from the user's workspace settings.

## Real RTX 4090 checks

Accelerate used `--gpu_ids 0 --num_processes 1`; PyTorch device 0 was verified as the RTX 4090 before loading models. Existing local checkpoints, cached caption embeddings and AntelopeV2 models were used. Original models, datasets and LoRAs were not overwritten.

Both face smokes used the full compact ConvRot INT8 H3 transformer, real video VAE and real AntelopeV2 reward, two optimizer updates, 128px resolution, two denoising steps, DRaFT K=1, 48 swapped blocks and VAE checkpointing. Preview generation was disabled for timing. Initial LoRA and seeds were identical between modes.

| Training mode | First update | Second update | Peak allocated VRAM | Final gradient norm | Saved tensors |
|---|---:|---:|---:|---:|---:|
| One frame | 11.09 s | 6.12 s | 9.91 GiB | 0.0836 | 600, all finite |
| Native five frames | 10.83 s | 6.78 s | 10.49 GiB | 0.1757 | 600, all finite |

At this small size, the second five-frame update took about 11% longer and allocated about 0.58 GiB more. Other applications occupied GPU memory during the test, and only two updates were measured; these are smoke measurements, not isolated performance benchmarks or full-resolution VRAM estimates.

Neither tiny two-step smoke produced a detector-quality generated face. The real reward's missed-detection fallback remained differentiable in all frames. These checks demonstrate finite training and checkpoint output, not identity improvement. A realistic resolution, denoising schedule and longer quality comparison remain the final user check. One-frame mode remains the default and explicit low-memory fallback.

A real compact-H3 latent-cache CLI test used the video VAE's FP32 encoder. The missing cache was written, the next matching run retained exactly the same file timestamp and provenance, and changed source pixels rebuilt the cache with a new timestamp and identity. The resulting latent tensors were finite. Caption and native video/audio invalidation were covered by the automated contract tests.

## Runtime and artifacts

Checks used the existing Windows venv: Torch 2.6.0+cu124, Transformers 4.57.6, Accelerate 1.6.0, Safetensors 0.4.5 and PyAV 14.0.1. They do not establish compatibility with the newer dependency pins in `pyproject.toml`; no dependencies were changed for this task.

Private smoke logs, performance JSON, test settings and saved LoRAs are retained under `.private/future_ideas_validation/`. Nothing was committed or pushed.
