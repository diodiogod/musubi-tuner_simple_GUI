# Experimental H3 automatic block memory

In **Memory & runtime**, choose **Block Memory Management → Automatic (experimental)**.
Classic GUI exposes the same choice under memory settings. Existing recipes remain
on **Fixed blocks (existing)** and retain their saved Blocks to Swap value.

Automatic mode measures complete training microbatches, including backward,
auxiliary losses and optimizer work. It starts unfamiliar image/video/text shapes
conservatively, then gradually keeps more frozen blocks on the GPU when there is
room. Larger batches can cause it to stream more blocks again. It never changes
the dataset's resolution, frame count, captions, or quality-protection settings.

Start with a **2 GiB safety margin**, minimum **2** and maximum **48** streamed
blocks. These are experimental defaults, not a promise that every clip fits.
Swapping saves weight memory, not all activation/attention memory. Reduce resolution
or frame count when even maximum swapping cannot fit the training calculations.
For demanding mixed-video buckets, try a larger margin (for example 4 GiB) if
Windows shows shared-GPU-memory usage or training becomes unexpectedly slow.

Gradient checkpointing is required; Torch Compile is not currently supported in
automatic mode. Automatic uses H2D-only frozen-weight streaming and additional
system RAM for immutable masters of all blocks. The pruned INT8 model needs about
18 GiB for these masters alone, in addition to other process allocations. Pageable
masters are the default; pinned mode can substantially increase pinned/shared memory.

## Diagnostics and saved runs

`Auto memory` log lines explain transitions and show driver memory separately from
PyTorch allocations. Tracker metrics `memory/auto_streamed_blocks`,
`memory/auto_torch_peak_gib` and `memory/auto_microbatch_seconds` report each measured
microbatch; the timing includes residency transitions. PyTorch peak allocation is
**not** total physical card usage.
Under physical memory pressure, the controller releases unused allocator cache at
a safe boundary and caps reclaimable cache to physical capacity. This can reduce
cache-related pressure, but does not guarantee Windows will never page GPU memory.

Runtime profiles are process-local. Resuming starts memory learning conservatively;
profiles do not become model weights or required checkpoint artifacts. LoRA metadata
records the automatic configuration. Preview memory is excluded from training
measurements and preview entry releases dedicated block residency.

Training OOMs are not automatically replayed: partial backward can already have
changed accumulated gradients. This feature is experimental; short smoke tests do
not establish long-run convergence or speed gains for every dataset.

## Validation status

Focused RTX 4090 checks have exercised native pruned ConvRot training with
Dynamic Sigma on every step, including 90-frame 512×512 and 22-frame 1024×1024
synthetic cached batches. The mixed run reached eight steps and saved its LoRA
and optimizer state. Reloading that full state completed two additional steps
and saved successfully. Small CUDA tests check gradients against a resident model,
accumulation, both transfer engines, and preview/training transitions.
A higher-pressure alternating run (90 frames at 768px and 22 frames at 1024px)
also completed eight steps and saved. It demonstrated 46→48 demotion before the
larger-working-memory batch and 48→44 promotion for the cheaper batch. This test
also exposed retained allocator-cache pressure; the graph-boundary cache guard
reduced the following cheaper-bucket time from about 56 seconds in the earlier
probe to about 31 seconds. Those isolated timings are diagnostic, not a benchmark.
Compact still-image training also completed six synthetic steps across 256px and
512px buckets with every-step Dynamic Sigma, promoted residency from 48 to 46
streamed blocks, and saved the LoRA successfully.
The compact scheduled-preview lifecycle was exercised with a real FP16 video VAE:
five-frame, two-denoising-step synthetic previews were decoded and written between
training updates, followed by another successful backward/optimizer update. This
checks residency restoration, not visual quality (the prompt embeddings were synthetic).

These are runtime smoke tests, not training-quality comparisons. They do not yet
establish a general speed advantage over a well-tuned fixed count.

A short matched 8-step compact run (256/512px synthetic buckets, rank 16,
BF16 ConvRot backward, AdamW8bit, every-step Dynamic Sigma) measured 61.22 seconds
of complete microbatch work in automatic mode versus 60.79 at fixed 48 blocks.
Excluding the first two warm-up steps, medians were 6.04 versus 6.10 seconds.
Automatic reached 42 streamed blocks; its last two steps averaged 5.77 seconds
versus 6.12 fixed. Timings include automatic transitions, but exclude model loading
and final saving. This is effectively a tie overall, not evidence of a general
speedup; longer and higher-pressure mixed workloads need further measurement.

## Implementation reference

ComfyUI's dynamic residency/prefetch architecture informed the design, but this is
an independent extension of Musubi's H2D streaming implementation, with no ComfyUI
runtime dependency or copied GPL implementation. Residency changes happen between
completed microbatches, never midway through a live training graph.
