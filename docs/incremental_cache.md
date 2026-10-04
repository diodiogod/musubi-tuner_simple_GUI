# Incremental cache preparation

Both GUIs default to **Only update missing or outdated cache entries**. Enable image and/or caption cache preparation to discover new or changed dataset entries. Normal and staged runs use the same policy. Turn the incremental option off for a deliberate full rebuild.

The existing **Fully rebuild caption cache before every stage** control remains a deliberate text-only full rebuild and overrides incremental reuse for those stage caption commands.

Wan, Flux.2, Krea 2 and compact MiniMax H3 record provenance inside their safetensors caches. Latent identity covers decoded source/control pixels, resolution, video/audio windows and VAE settings. Caption identity covers caption text, encoder/tokenizer settings and precision; visual teacher caches also cover decoded images and controls. Existing posterior-policy and DOP validators still apply. Model files are identified by resolved path, size and modification time, so replacements at the same path invalidate caches. Moving a model or changing its timestamp can conservatively rebuild entries.

Older caches without provenance rebuild once during cache preparation. Missing, stale or unreadable entries rebuild; matching entries retain their files and timestamps. Removing entries keeps the existing cache-cleanup policy. This does not automatically scan datasets when cache preparation is off.

Native MiniMax H3 reuses its existing checkpoint/media/presentation metadata contract, including cache precision, crop position, one-frame controls and teacher conditions. Its commands use `--skip_existing` without the extra generic provenance flag.

The shared CLI's existing `--skip_existing` behavior is unchanged unless `--validate_cache` is also supplied. GUI commands supply both for incremental preparation, and `--validate_cache` alone for full rebuilds, so a subsequent incremental run can reuse the freshly built caches.

Tracking idea: [issue #2](https://github.com/diodiogod/musubi-tuner_simple_GUI/issues/2).
