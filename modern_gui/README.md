# Musubi Studio — experimental web GUI

Musubi Studio is a parallel local web interface for this Windows-focused
Musubi Tuner fork. It is visually inspired by the workflow clarity of Ostris
AI Toolkit, but it does not run AI Toolkit's trainer, configuration schema, or
queue worker. Existing modules in `backends/` remain the authority for Musubi
cache and training commands.

The original Tkinter application is unchanged and remains available through
`LAUNCH_GUI.bat`. The web interface can also open it from the sidebar for
specialized inspection or comparison during the experimental period.

## Start

Run:

```text
LAUNCH_MODERN_GUI.bat
```

The launcher uses `venv\Scripts\python.exe` when present and opens
`http://127.0.0.1:8675`. The server binds to localhost by default and has no
remote-access mode in the launcher.

## Current capabilities

- model-family-first dashboard and guided five-step recipe;
- concise label-hover help with clickable documentation only for deeper topics;
- searchable, mode-aware visual controls for every scalar setting currently
  stored in `last_settings.json`;
- complete JSON editor that preserves unknown and future settings;
- structure-preserving TOML parsing, optimistic conflict detection, and atomic
  saves using `tomlkit`;
- inherited-aware image/video source defaults, Directory/JSONL switching,
  ordering, safe duplication, controls, video extraction settings, and
  one-click expansion of valid immediate media subfolders into separate TOML
  sources; source rows can also be disabled and restored without deleting
  their settings (disabled entries are kept as explicit GUI-owned comments);
- paginated real-media browsing with lazy image/video previews, search,
  training-eligibility filters, and repeat-weighted epoch totals;
- explicit atomic sidecar and JSONL caption editing with stale-write
  protection, plus Musubi-parity resolution, readability, caption, nested-file,
  aspect-ratio, control, and effective-sample audits;
- card-based, mode-aware sample-prompt planning with resolution presets,
  inline readiness checks, local preview state, and a reusable global library;
- a connected stage timeline with effective output names, typed state/LoRA
  handoffs, cache policy, stage-specific validation, and dedicated editors;
- user notes plus the same generated settings summary that is attached to the
  actual run;
- Krea batch previews plus one-at-a-time MiniMax H3 standalone previews and automatically refreshed sample outputs;
- dedicated Krea/MiniMax face-refinement setup, reference analysis/review, and pose plan,
  plus Krea-only fixed Turbo evaluation and result galleries;
- Musubi command previews through the existing backend adapters;
- normal and typed staged process execution with bounded live logs;
- explicit standard-stage handoffs: preserve the complete optimizer/scheduler state, or load the prior LoRA into a fresh optimizer when a stage intentionally changes learning rate, optimizer, scheduler, warmup, or timestep sampling;
- face-stage LoRA handoffs;
- live step, epoch, loss, depth-anchor, DOP, GPU, and VRAM monitoring;
- loss graph, wipe/side-by-side comparison, keyboard/touch navigation, and
  ordinary sample gallery;
- merged desktop/web job history with discovery, details, repeat/edit,
  parameter/prompt import, filesystem actions, continuation, and recovery;
- LoRA conversion, Accelerate setup, native local path selection, concept
  workspace layout, adapter-size estimation, and settings import/export/reset;
- additive continuation and validated exact failed-run recovery.

The maintained [parity tracker](PARITY.md) maps the classic GUI capabilities to
their modern locations and records the invariants used by regression tests.

## Safety and compatibility

- Ordinary continuation never enables `--resume_exact_position`.
- Exact recovery requires a failed/stopped normal job, a complete settings
  snapshot, nonempty model/optimizer/scheduler/RNG state files, and a numbered
  epoch or step state folder.
- A full-state standard handoff requires the preceding state directory. A fresh-optimizer handoff requires the preceding completed LoRA; existing plans default to full-state behavior.
- Face-stage handoff requires the preceding LoRA and verifies the expected
  refined LoRA before advancing.
- Included sample-prompt cards are validated and atomically materialized into
  the exact prompt snapshot passed to Musubi; command previews remain
  filesystem-read-only.
- Staged History keeps the base recipe plus final-stage lineage, so ordinary
  continuation resumes the last successful state or refined LoRA without
  enabling positional recovery.
- Dataset visual edits patch the canonical TOML document instead of rebuilding
  it. Comments, order, inherited fields, unknown keys, and specialized trainer
  options are retained, and a file changed externally is not overwritten.
- Dataset previews use short-lived opaque tokens rather than a generic local
  file endpoint. Caption writes are local-only, size-bounded, explicit, and
  atomic.
- Sample image serving is restricted to configured output directories.
- State-changing HTTP actions require a loopback client and, when a browser
  sends an Origin header, the exact local application origin.
- The service strips inherited distributed-launch variables before starting
  Musubi, matching the desktop GUI's single-process Windows behavior.

No generated files or settings are committed automatically. The web and
desktop interfaces currently share `last_settings.json`; saving in either
interface intentionally updates the workspace used by the other.
