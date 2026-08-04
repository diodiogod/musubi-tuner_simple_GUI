# Modern GUI parity tracker

This is the implementation contract for the experimental web interface. A feature is
`complete` only when its workflow is visually usable, uses the existing Musubi backend
semantics, and has regression coverage. Merely retaining a value in raw JSON is not parity.

| Classic workspace / capability | Modern location | Status |
|---|---|---|
| Model family and model paths | Dashboard → New training → Model | Complete |
| Dataset TOML selection and visual builder | New training → Dataset; Datasets workspace | Complete |
| General inheritance, Directory/JSONL sources, ordering, controls, and video extraction | Datasets → Source settings | Complete |
| Paginated media/video preview, eligibility filters, and repeat weighting | Datasets → Media / Health | Complete |
| Explicit conflict-safe sidecar and JSONL caption editing | Datasets → Media inspector | Complete |
| Comment/order/unknown-key-preserving canonical TOML edits | Datasets → TOML | Complete |
| New / weights continuation / exact recovery distinction | New training → Dataset; History | Complete |
| Training parameters and optimizer/scheduler | New training → Method; All settings | Complete |
| Notes and generated settings summary | Training Plan → Notes | Complete |
| Standard staged training plan | Training Plan → Stages | Complete |
| Stage DOP/depth overrides and cache policy | Training Plan → Stages | Complete |
| Krea face-refinement stages | Face Refinement; Training Plan | Complete |
| DOP, Krea weight noise, depth anchor | New training → Method → Regularization | Complete |
| Complete advanced/runtime/logging/sampling settings | All settings | Complete |
| Sample prompt add/edit/delete/duplicate/enable | Training Plan → Sample prompts | Complete |
| Global prompt library and Krea preview generation | Training Plan → Sample prompts | Complete |
| Prompt-library favorites, metadata editing, deletion, and tested thumbnails | Prompt library dialog | Complete |
| Automatic library-thumbnail capture after successful prompt tests | Active run; Prompt library | Complete |
| Sample comparison and gallery | Samples | Complete |
| Live counters, GPU, loss, depth/DOP, console | Active run | Complete |
| Standard/staged run and stop | Review; Active run | Complete |
| Command preview and cache sequencing | Recipe; Active run | Complete |
| Job status/history and continuation/recovery | History | Complete |
| Repeat/edit, apply portable parameters, import prompts | History | Complete |
| Load a completed Krea job into face refinement | History → Refine face identity | Complete |
| Face refinement setup/preflight/pose plan | Face Refinement | Complete |
| Face model download, reference exclusion, and manual pose correction | Face Refinement → Setup / Analyze | Complete |
| Pose presets, per-pose stopping goals, prompt ideas, and prompt import | Face Refinement → Refine | Complete |
| Evaluation-driven weak-pose plan and baseline deltas | Face Refinement → Evaluate | Complete |
| Fixed Turbo face evaluation and result galleries | Face Refinement | Complete |
| LoRA format conversion | Tools | Complete |
| Accelerate configuration | Tools | Complete |
| Load/save/reset settings files | All settings | Complete |
| Theme switching | Sidebar | Complete |
| Keyboard shortcuts and accessible interaction | Global | Complete |

## Invariants

- The desktop GUI remains available and unchanged while parity is developed.
- Continuation is additive; verified failed-run recovery alone is positional.
- Krea RAW training, optional Turbo sampling, projector behavior, FP8, DOP, weight
  noise, depth anchor, face refinement, and staged typed handoffs must not regress.
- Ordinary help is attached to the label on hover/focus. A visible question mark is
  reserved for substantial documentation that opens a dismissible dialog.
- Structured prompt cards are the execution source of truth. Enabled cards are
  validated and snapshotted atomically before Musubi starts; command previews do
  not write that snapshot.
- Stage cards expose the effective artifact name and handoff type. History keeps
  final-stage lineage while retaining the unchanged base recipe.
