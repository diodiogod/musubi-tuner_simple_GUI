# Differential Output Preservation (DOP)

DOP is an experimental LoRA-training option for **Krea 2** and **FLUX.2 Klein 4B/9B**. It helps a LoRA learn a named subject without replacing the base model's broader knowledge of the subject's class.

For example, if every caption contains `Alice` and the preservation class is `a woman`, the cache stores both:

- the normal caption: `portrait of Alice in a garden`
- the class caption: `portrait of a woman in a garden`

During each training step, the normal subject loss still runs. DOP additionally asks the base model, with the LoRA disabled, how the class caption should behave. It then penalizes the LoRA only when enabling it changes that generic-class answer too much.

This can help prompts distinguish the trained character from other people of the same class. It is not a face-similarity loss and does not guarantee better identity by itself.

## GUI setup

1. Select Krea 2 or FLUX.2 Klein.
2. Open **Training → Identity & Class Preservation · DOP**.
3. Enable DOP.
4. Enter the exact subject trigger used in every caption.
5. Enter a natural preservation phrase such as `a person`, `a woman`, `a man`, `a dog`, or `a car`.
6. Start with strength `1.0`.
7. Enable **Re-cache Text Encoder Outputs** for the first DOP run. Staged DOP runs later validate and reuse matching caches automatically, encoding only missing or changed captions.

Trigger matching ignores capitalization and replaces literal text, including multi-word triggers. Caching stops with a clear error if any caption lacks the trigger. The cache carries a signature of the trigger/class pair; training refuses stale or mismatched DOP embeddings.

Choose wording that makes the replaced caption read naturally. For example, `photo of Bob outdoors` should normally become `photo of a man outdoors`, not `photo of man outdoors`. Different caption grammar may require a different phrase.

## Cost and monitoring

DOP performs three DiT predictions per batch instead of one: normal subject training, a frozen base-model class prediction, and a LoRA-enabled class prediction. Expect a substantial speed cost. The base teacher runs without gradients, so the memory increase is smaller than running three normal training passes, but model and configuration differences still matter.

The Monitor page reports:

- **class error**: the raw difference between the LoRA-enabled and base-model class predictions;
- **added loss**: class error multiplied by the configured strength.

Finite, changing values confirm DOP is active. A falling value means the adapter is becoming less disruptive to the generic class, but it is not an image-quality score.

## Staged training

Every standard stage can open **DOP Settings…** and choose:

- **inherit**: follow the main DOP configuration;
- **enable**: force DOP on for that stage;
- **disable**: turn it off for that stage.

Strength, trigger, and class may be overridden per stage. Before each DOP stage, the GUI checks the caption, trigger/class signature, and required class embedding in every cache. Matching entries are reused; missing, changed, or incompatible entries are encoded again. **Fully re-cache text before every standard stage** is available for deliberate clean rebuilds, but is normally unnecessary. Face Refinement stages use their own loss and do not run DOP.

The main **Re-cache Text Encoders Before Training** checkbox stays off during this validated reuse pass. The stage log reports `Text cache: validate and reuse`; this is intentionally different from a full rebuild.

To reduce peak VRAM, the trainer finishes the normal/depth backward pass first, releases that graph, and then runs the DOP preservation backward pass. Both gradients accumulate into the same optimizer step, so this changes memory scheduling rather than the objective.

Staged plans containing Face Refinement also verify the optional AntelopeV2 detector and identity files before stage 1 starts, preventing a late failure after a long standard-training stage.

One practical experiment is stronger preservation in an early, lower-resolution stage and weaker or disabled preservation in a later identity/detail stage. This is not a universal recommendation; compare fixed prompts and seeds.

## Attribution

The technique was studied from [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit), where it is called Differential Output Preservation. This repository contains an independent Musubi adaptation for cached-text Krea 2 and FLUX.2 Klein training.
