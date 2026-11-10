"""MOSS settings shared by both interfaces."""
MODE = "MOSS-Image (Experimental)"
DEFAULTS = {
    "moss_dit_model": "", "moss_text_encoder": "", "moss_tokenizer_cache_dir": "",
    "moss_max_token_length": "4096", "moss_text_encoder_cpu_streaming": True,
    "moss_convrot_int8": False, "moss_convrot_bwd_mode": "bf16",
    "moss_lora_target_preset": "All main-block projections",
    "moss_sample_flow_shift": "5.0", "moss_sample_skip_layers": "22",
    "moss_sample_sampler": "euler", "moss_sample_scheduler": "official",
    "moss_sample_lora": "", "moss_sample_lora_multiplier": "0.6",
    "moss_pretraining_schedule": True, "moss_logit_mean": "-1.09", "moss_logit_std": "0.95",
    # Empty means inherit the legacy pretraining checkbox when loading old recipes.
    "moss_timestep_schedule": "",
    "moss_timestep_cutoff": "0.95", "moss_timestep_uniform_ratio": "0.01",
    "moss_depth_anchor_enabled": False, "moss_depth_anchor_weight": "0.0",
    "moss_depth_anchor_model": "depth-anything/Depth-Anything-V2-Small-hf",
    "moss_depth_anchor_input_size": "518", "moss_depth_anchor_gradient_weight": "0.5",
    "moss_depth_anchor_grad_checkpoint": True, "moss_depth_anchor_vae_device": "training",
    "moss_keep_depth_helpers_on_gpu": False,
    "moss_weight_noise_enabled": False, "moss_weight_noise_sigma": "0.0", "moss_weight_noise_mode": "relative",
    "moss_weight_noise_bound_norm": False,
    "moss_repa_enabled": False, "moss_repa_weight": "0.5",
    "moss_repa_encoder": "facebook/dinov2-small", "moss_repa_block": "8",
    "moss_repa_image_size": "518", "moss_repa_device": "training", "moss_repa_cache_size": "64",
    "moss_repa_cache_dir": "",
    "moss_ema_enabled": False, "moss_ema_decay": "0.98", "moss_ema_device": "training",
    "moss_automagic3_min_lr": "0.00005", "moss_automagic3_max_lr": "0.001",
    "moss_fast_lora": False, "moss_text_pad_multiple": "0",
    "moss_compile_checkpoint": "inside", "moss_sdpa_backend": "default",
    "moss_clean_end_share": "",
    "moss_unpadded_batches": False,
}
PERFORMANCE_KEYS = ("moss_fast_lora", "moss_unpadded_batches", "moss_text_pad_multiple", "moss_compile_checkpoint", "moss_sdpa_backend")
MUON_ARGS = 'adjust_lr_fn="match_rms_adamw" weight_decay=0.01 momentum=0.95 nesterov=True ns_steps=5'
CLEAN_END_DETAILS = (
    "Every training step takes one of your pictures and adds noise to it. MOSS practices removing that noise. "
    "This setting chooses how often it gets a clearer picture versus a much noisier one. It uses the same MOSS "
    "model throughout; there are no separate high-noise and low-noise models.\n\n"
    "Fixed 40 means roughly 40 out of every 100 examples have less than half noise, and the remaining 60 have "
    "more noise. This balance stays the same for the entire run. Clearer examples practice smaller corrections "
    "to an already-visible picture; noisier examples require reconstructing more of it. More of either is not "
    "automatically better. Blank keeps the model's original distribution, about 11% less-noisy examples.\n\n"
    "Fizgig also offers a changing balance. Its current Muon preset starts with 30% less-noisy examples, moves "
    "to 60% after epoch 7, and occasionally revisits the noisier balance at a lower learning rate. This is an "
    "experimental recipe based on its author's dataset/optimizer results, not a requirement of either app or "
    "a universal recommendation. A fixed 40 is a valid comparison setting.\n\n"
    "Advanced: paste 30:7:60 for 30% through epoch 7 and 60% afterward. 30:7:60:8 slides to 60% by epoch 8. "
    "30:7:60:8:3 then uses 30% every third epoch after epoch 8, at half the optimizer learning rate, and 60% "
    "on the other epochs. That five-part refresh needs a fixed-rate optimizer such as Muon/AdamW. "
    "This only applies to MOSS pretraining. It needs no recaching and adds no extra model pass."
)
PERFORMANCE_DETAILS = (
    "Want to try the speed techniques used by Fizgig for MOSS? Choose Muon to batch updates to same-shaped "
    "adapter matrices. Its defaults match Fizgig: learning rate 0.0001, weight decay 0.01, momentum 0.95, "
    "Nesterov on, 5 orthogonalization iterations, and match_rms_adamw scaling. You may override them in "
    "Optimizer Args. Muon works on our existing PyTorch runtime. Switch optimizers by continuing from LoRA "
    "weights with a fresh optimizer, rather than resuming an AdamW saved state.\n\n"
    "Fast LoRA compute keeps trainable weights and optimizer state in FP32, but calculates the LoRA in BF16 "
    "and saves smaller activations for backward. It follows Fizgig's low-precision LoRA path. It changes rounding; "
    "it is an optional alternative, not an identical numerical result. Plain LoRA is supported; leave it off for "
    "LoKr/LoHa. Dropout falls back to the ordinary path. No recaching is required.\n\n"
    "Unpadded image passes follows Fizgig's current MOSS approach: each image runs with its own true caption "
    "length and no padding mask. A larger batch still averages the image losses before the same optimizer "
    "update. This can avoid a slower masked-attention kernel and wasted caption work. Leave caption padding "
    "at 0 for this path. Alternatively, caption padding rounds a masked batch to a token multiple (try 16), "
    "reducing compiler shapes at the cost of extra tokens. It is a separate experiment, not Fizgig's MOSS default.\n\n"
    "Compile is the existing shared toggle. MOSS's fast Inductor path requires the optional performance runtime: "
    "PyTorch 2.15, matching torchvision and Triton. Install with python tools/bootstrap_environment.py "
    "--performance-runtime. Existing launchers keep using the same environment. Compile needs zero swapped "
    "blocks and ordinary GPU gradient checkpointing. Inside compiles the checkpoint too (Fizgig's MOSS choice); "
    "Outside leaves checkpointing eager and may fit higher resolutions more easily. New shapes take time to compile.\n\n"
    "cuDNN preferred is an optional SDPA backend preference, with other PyTorch kernels available as fallbacks. "
    "Its initial shape planning can be slow. It applies only to MOSS SDPA, not Sage/Flash attention. These options "
    "do not turn off DOP, depth, REPA or EMA; those still add their own training cost. Actual speed requires a GPU "
    "measurement. Matching these options alone does not reproduce another trainer's entire recipe or RNG sequence."
)
OPTIMIZER_DETAILS = (
    "Muon is orthogonalized momentum, with same-shaped matrix updates batched as in Fizgig. Use the dedicated "
    "Muon choice to get its defaults (match_rms_adamw scaling, weight_decay=0.01, momentum=0.95, nesterov=True, "
    "ns_steps=5). Start with learning rate 0.0001; the ordinary LR scheduler still applies. It is available without "
    "upgrading PyTorch.\n\n"
    "Automagic4 is Ostris's experimental extension of Automagic3. It adapts its rate and reduces updates to "
    "sensitive weights. Start at 0.000001, constant scheduler, zero warmup. Optional Optimizer Args include "
    "min_lr, max_lr, sensitivity_quantile=0.9 and step_target=0.1. The MOSS Automagic3 bound fields apply only to "
    "Automagic3. Version 4 does not inherit those limits. Neither version guarantees better quality.\n\n"
    "AdamW can also use fused=True in Optimizer Args on CUDA. This fuses AdamW's optimizer step; it does not "
    "update weights during backward. Changing optimizer requires a fresh optimizer state. All existing "
    "optimizer choices remain available. No latent or caption recaching is needed."
)
CHOICES = {
    "moss_compile_checkpoint": ["inside", "outside"],
    "moss_sdpa_backend": ["default", "cudnn preferred"],
    "moss_timestep_schedule": ["MOSS pretraining (default)", "Sigmoid + High (AI Toolkit)", "General timestep controls"],
    "moss_ema_device": ["training", "cpu"],
    "moss_sample_sampler": ["euler", "ddim"],
    "moss_sample_scheduler": ["official", "simple"],
    "moss_convrot_bwd_mode": ["bf16", "int8"],
    "moss_lora_target_preset": ["All main-block projections", "Attention only"],
    "moss_depth_anchor_vae_device": ["training", "secondary", "cpu"],
    "moss_weight_noise_mode": ["relative", "absolute"],
    "moss_repa_device": ["training", "cpu"],
}
PATHS = {"moss_dit_model", "moss_text_encoder", "moss_tokenizer_cache_dir", "moss_repa_cache_dir", "moss_sample_lora"}
REPA_LABELS = {
    "moss_depth_anchor_enabled": "Enable structural depth guidance",
    "moss_weight_noise_enabled": "Enable adapter weight noise",
    "moss_unpadded_batches": "Unpadded image passes (Fizgig-style)",
    "moss_clean_end_share": "Less-noisy training examples (%)",
    "moss_fast_lora": "Fast BF16 LoRA compute (Fizgig-style)",
    "moss_text_pad_multiple": "Caption padding multiple (0 = off)",
    "moss_compile_checkpoint": "Checkpoint placement when compiling",
    "moss_sdpa_backend": "MOSS SDPA backend preference",
    "moss_timestep_schedule": "Training noise schedule",
    "moss_ema_enabled": "Use EMA smoothed checkpoints (optional)",
    "moss_ema_decay": "EMA smoothing", "moss_ema_device": "Store averaged adapter on",
    "moss_automagic3_min_lr": "Automagic3 minimum learning rate",
    "moss_automagic3_max_lr": "Automagic3 maximum learning rate",
    "moss_sample_sampler": "Default preview sampler",
    "moss_sample_scheduler": "Default preview noise schedule",
    "moss_sample_lora": "Sampling-only Turbo LoRA (optional)",
    "moss_sample_lora_multiplier": "Sampling-only Turbo strength",
    "moss_repa_enabled": "Use REPA visual guidance (experimental)",
    "moss_repa_weight": "REPA strength", "moss_repa_encoder": "DINOv2 model or local folder",
    "moss_repa_block": "MOSS layer to guide", "moss_repa_image_size": "Visual teacher image size",
    "moss_repa_device": "Teacher compute device", "moss_repa_cache_size": "Remembered images (RAM)",
    "moss_repa_cache_dir": "REPA feature cache folder",
}
TIMESTEP_DETAILS = (
    "Optional Clean-end share matches Fizgig's noise-distribution adjustment. Blank keeps our default. "
    "40 means about 40% of training examples below half noise. 30:7:60 uses 30% for epochs 1-7, then 60%. "
    "30:7:60:8 slides from 30% at epoch 7 to 60% at epoch 8. Adding :3 keeps the higher share afterward "
    "and uses the lower share every third epoch after the slide, at half the optimizer learning rate. "
    "The five-part refresh needs a fixed-rate optimizer such as Muon or AdamW. The share replaces Logit mean "
    "while keeping standard deviation, cutoff and uniform ratio. It only applies to MOSS pretraining.\n\n"
    "This chooses how noisy the training examples are. At high noise, the model must recover the overall "
    "subject and composition; at low noise, more of the picture is visible and it practices finer corrections. "
    "A schedule changes how often each difficulty is practiced. It does not change the dataset resolution.\n\n"
    "MOSS pretraining keeps our existing default, based on MOSS's published training distribution. "
    "Its advanced values remain editable below. Older recipes keep this behavior unless they had explicitly "
    "turned it off, in which case they keep General timestep controls.\n\n"
    "Sigmoid + High follows the combination used in the community AI Toolkit MOSS recipe. Sigmoid creates "
    "a spread of noise levels; High then favors the noisier end. This is an alternative distribution, not "
    "an extra quality-protection method or a promise of better results. Our default already favors noisy "
    "examples, but the two distributions are different. It works with LoRA, LoKr, EMA and Automagic3; "
    "none of those other options is required or changed automatically.\n\n"
    "General timestep controls uses the shared Timestep Sampling and Flow Shift fields instead. "
    "Those fields and the pretraining-specific values are ignored while Sigmoid + High is selected. "
    "Switching schedules needs no latent or caption recaching and adds no extra model pass. "
    "Use the same schedule when reproducing a run; changing it on a continuation changes the training recipe."
)


def timestep_schedule(settings):
    """Resolve explicit selection, or preserve an older recipe's checkbox."""
    value = settings.get("moss_timestep_schedule")
    if value is None or value == "":
        return CHOICES["moss_timestep_schedule"][0 if settings.get("moss_pretraining_schedule", True) else 2]
    if value not in CHOICES["moss_timestep_schedule"]:
        raise ValueError("Choose a valid MOSS Training noise schedule.")
    return value


def with_timestep_schedule(settings):
    result = dict(settings)
    result["moss_timestep_schedule"] = timestep_schedule(settings)
    result["moss_pretraining_schedule"] = result["moss_timestep_schedule"] == CHOICES["moss_timestep_schedule"][0]
    return result


EMA_DETAILS = (
    "Want smoother samples and saved LoRAs while training continues normally? EMA keeps a moving average of "
    "your adapter's recent weights. With smoothing 0.98, each update keeps 98% of the previous average and "
    "mixes in 2% of the latest weights. It is compatible with LoRA and LoKr and does not average the frozen base model.\n\n"
    "Start with 0.98 to try the community MOSS recipe. Higher values respond more slowly and can lag behind "
    "new details; lower values follow recent learning more closely. EMA smooths weight changes. It is not an "
    "extra quality-preservation loss, and it cannot prevent overtraining or fix bad training data.\n\n"
    "When enabled, scheduled samples and ordinary exported LoRA files use the averaged weights. Full saved "
    "states retain the live training weights, optimizer and EMA average separately, so exact resuming remains "
    "consistent. A saved-state preview uses its ema.safetensors companion. Keep EMA enabled with the same smoothing "
    "when resuming a full state. To turn EMA on/off or change smoothing between runs, continue from LoRA weights "
    "with a fresh optimizer instead. No image or caption recaching is needed.\n\n"
    "Training stores the extra adapter copy on the main GPU for speed. CPU stores it in system RAM but transfers "
    "adapter weights after each update, which can slow large full-rank LoKr runs. EMA adds an FP32 adapter-sized "
    "copy; it does not require another full diffusion model."
)
AUTOMAGIC3_DETAILS = (
    "Automagic3 is an experimental optimizer from Ostris's AI Toolkit. Like Prodigy, it adapts learning rates "
    "automatically, but it uses a different rule: consistent update directions encourage a larger step; "
    "repeated direction reversals encourage a smaller one. It also scales and clips updates. It is not a "
    "guarantee of better LoRAs.\n\n"
    "Learning rate is its starting point. The shared community MOSS recipe starts at 0.0005 with minimum "
    "0.00005 and weight_decay=0.00001. A gentler starting point is 0.0001. Our MOSS controls expose the minimum "
    "and maximum rates; the minimum can stop learning from fading, but forcing it too high can prevent the "
    "optimizer from settling. These are experiments, not proven best settings.\n\n"
    "Use the constant scheduler with zero warmup: Automagic3 controls the effective rate itself, and the LR "
    "logs report that adapted rate. Musubi uses normal step-time updates (fused=False), preserving gradient "
    "accumulation, clipping and separate DOP backward passes. Trainable adapters stay FP32; mixed-precision "
    "BF16 for the base model remains supported. No PyTorch upgrade is required.\n\n"
    "Changing optimizer requires a new optimizer state: continue from LoRA weights, not a full saved state "
    "created with AdamW. Caption and latent caches can be reused. The optimizer does not change the noise "
    "schedule, rank, EMA or other regularization settings."
)
REPA_METRICS_HELP = (
    "REPA compares features learned by MOSS with features from a frozen visual teacher (DINOv2). "
    "These numbers describe that agreement for the latest logged training batch.\n\n"
    "REPA loss: lower means closer agreement with the teacher. Added loss: the REPA loss multiplied by "
    "your REPA strength; this is the amount added to the training objective. For example, loss 0.54 "
    "at strength 0.5 adds 0.27.\n\n"
    "Similarity: higher means closer agreement, on a scale from -1 to 1. REPA loss is 1 minus this similarity. "
    "It is not an image-quality score or a percentage of correct details. A better number alone does not "
    "prove that your LoRA looks better or is protected against overtraining; compare generated samples too. "
    "A dash means no value has been reported."
)
REPA_DETAILS = (
    "Want to give MOSS an extra visual teacher while it learns your pictures? REPA adds guidance from DINOv2, "
    "a frozen image-understanding model. Normal training still teaches your captioned images. REPA also encourages "
    "an early MOSS layer to recognize visual features that agree with the teacher.\n\n"
    "This follows the REPA research and SimpleTuner's MOSS image approach. It is a learning experiment, not a promise "
    "against overtraining, lost knowledge or poor image quality. DOP compares the base model's outputs; depth guidance "
    "compares structure. They are separate options and can be combined, but try REPA alone first to judge it.\n\n"
    "Start with strength 0.5, layer 8 and image size 518. Our default uses DINOv2 Small to reduce memory; "
    "SimpleTuner's example uses the much larger Giant encoder. These are experimental starting values, "
    "not a measured MOSS quality recommendation. The encoder downloads on first use, or enter a local Hugging Face "
    "DINOv2 model folder. No new dataset labels or latent/text recaching are required.\n\n"
    "Before the training model loads, REPA prepares and saves the teacher features for your dataset. The first "
    "preparation can take time, especially at 2K. Later epochs and runs read those saved features, even when they "
    "are no longer in the small RAM cache. Changed images, VAE or teacher settings prepare new features automatically. "
    "REPA does not repeatedly decode images or load its teacher during training.\n\n"
    "GPU compute is normally faster for preparing features. CPU avoids helper VRAM but can be very slow. "
    "High-resolution VAE decoding can still need substantial memory during preparation. Depth, if enabled, "
    "still decodes the current prediction during training; REPA caching does not remove that separate cost.\n\n"
    "REPA works with the MOSS BF16 and ConvRot training paths. It adds a small learned projection head to full "
    "resume states; generation LoRAs need no DINO or extra head. To turn REPA on/off or change its encoder, layer "
    "or image size between runs, continue from LoRA weights with a fresh optimizer.\n\n"
    + REPA_METRICS_HELP
)
SAMPLE_DETAILS = (
    "Want previews to use the same Turbo LoRA you use for generation? Select that MOSS LoRA here. "
    "It turns on only while generating a standalone preview or a scheduled training sample, then turns off before learning continues. "
    "It is never merged into the training model or exported inside your trained LoRA.\n\n"
    "A useful experimental recipe is 25 steps, DDIM, Simple schedule, CFG 3, flow shift 5 and Turbo strength 0.6. "
    "The prompt card's settings control both standalone and scheduled samples. Existing prompts keep their values until you apply the recipe. "
    "ComfyUI uses Euler updates for text-to-image DDIM; Simple selects different noise levels from MOSS's Official schedule.\n\n"
    "Your trained LoRA has a separate strength, normally 1.0. Turning off Current LoRA or clicking Generate base cover removes that trained LoRA, "
    "while keeping the optional Turbo adapter. This gives a fair before/after comparison with the same generation setup. "
    "Clear the Turbo file too if you want a completely unmodified base-model result.\n\n"
    "Generate base cover works before training: only the model, VAE and text encoder paths are needed. Covers are stored separately for each selected checkpoint; "
    "the library falls back to an earlier example until that model has one. H3 video covers use the first decoded frame. "
    "This uses the selected checkpoint throughout generation; it does not switch between SFT and Base models."
)
HELP = {
    "moss_depth_anchor_enabled": "Turn structural depth guidance on or off. Off adds no depth loss, regardless of the saved strength below. Turning it on first fills a starting strength of 0.01. Depth can add substantial time and VRAM use.",
    "moss_weight_noise_enabled": "Turn adapter weight noise on or off. Off adds no noise, regardless of the saved strength below. Turning it on first fills 0.0125 as an experimental starting value; this is not a proven MOSS optimum.",
    "moss_unpadded_batches": "Process each image with its own caption length and no padding mask, as current Fizgig MOSS does. Batch losses still combine into the same update. Keep Caption padding at 0; click ? for details.",
    "moss_clean_end_share": "How often MOSS practices with a clearer version of a training image. 40 means about 40% of examples have less than half noise; the other 60% are noisier. A single number stays fixed for the whole run. Blank keeps the original model schedule (about 11%). Click ? for Fizgig's optional changing schedule.",
    "moss_fast_lora": "Keep FP32 trainable weights while calculating plain LoRA in BF16, using Fizgig's smaller-activation approach. Optional; click ? for requirements and tradeoffs.",
    "moss_text_pad_multiple": "0 keeps exact caption lengths and is required for Unpadded image passes. Alternatively try 16 with masked batches to reduce compiler shape changes. Extra padding is masked; no recaching needed.",
    "moss_compile_checkpoint": "Inside follows Fizgig's MOSS compilation placement. Outside may reduce activation memory at higher resolutions. Used only when Compile is on.",
    "moss_sdpa_backend": "Default leaves PyTorch's choice unchanged. cuDNN preferred tries cuDNN with other SDPA backends available as fallbacks; new shapes take time to plan.",
    "moss_timestep_schedule": "Keep MOSS pretraining for the existing default, or try the community Sigmoid + High recipe. This changes which noise levels are practiced. No recaching is needed. Click ? for the differences.",
    "moss_ema_enabled": "Average recent adapter weights for smoother samples and LoRA exports. Optional and off by default. This is weight smoothing, not quality protection. Click ? for memory and resume details.",
    "moss_ema_decay": "Try 0.98. Each update mixes 2% of the latest adapter into the previous average. Higher smoothing reacts more slowly and may lag behind new details. No recaching is needed.",
    "moss_ema_device": "Training keeps the averaged adapter on the main GPU for speed. CPU saves that VRAM but copies adapter weights every optimizer update and can slow large LoKr runs. The frozen base model is not duplicated.",
    "moss_automagic3_min_lr": "Lowest rate Automagic3 may choose. 0.00005 follows the shared community recipe, not a proven optimum. Lower it if updates will not settle; a high floor can force continued changes. Use this field rather than min_lr in Optimizer Args.",
    "moss_automagic3_max_lr": "Highest rate Automagic3 may choose. 0.001 is a conservative ceiling for this experiment. It must be at least your starting Learning Rate. Use this field rather than max_lr in Optimizer Args.",
    "moss_sample_sampler": "Fallback when a prompt has no sampler of its own. A prompt card's sampler takes priority for standalone and scheduled samples. DDIM + Simple is the 25-step recipe; Euler + Official keeps the original behavior.",
    "moss_sample_scheduler": "Fallback when a prompt has no schedule of its own. A prompt card's schedule takes priority. Simple matches ComfyUI's noise levels; Official keeps MOSS's original schedule. Does not change training noise or require recaching.",
    "moss_sample_lora": "Optional MOSS Turbo LoRA used only for previews and scheduled samples. It is disabled during learning. Try strength 0.6 with 25 steps, DDIM + Simple and CFG 3. Click ? for the full explanation.",
    "moss_sample_lora_multiplier": "Strength of the optional sampling-only Turbo LoRA. Try 0.6 for the 25-step recipe. This does not change the trained LoRA's strength or training. Keep this setting the same when comparing base and trained samples.",
    "moss_repa_enabled": "Add an image-understanding teacher to normal MOSS training. Experimental and off by default. Click ? for how it works, memory costs and resume advice.",
    "moss_repa_weight": "How strongly the visual teacher guides learning. 0.5 is an experimental starting point. Too much can compete with the details you want to learn; disable REPA with its switch.",
    "moss_repa_encoder": "Hugging Face DINOv2 model ID or local model folder containing config and weights. Small downloads automatically and uses less memory. Base, Large and Giant cost more; SimpleTuner's example uses facebook/dinov2-giant. Other encoder families and register-token variants are not supported.",
    "moss_repa_block": "Which MOSS layer receives the extra guidance, counting from 0 to 41. Start with 8, following SimpleTuner's MOSS recipe. This does not skip or freeze the other layers.",
    "moss_repa_image_size": "Size of the picture shown to DINOv2, not your dataset resolution. 518 follows the reference recipe. 224 is a cheaper experiment. Use a positive multiple of 14.",
    "moss_repa_device": "Where REPA prepares its saved features before the training model loads. Training uses the main GPU; CPU avoids helper GPU memory but can be very slow. Once features are ready, REPA reads them without loading the teacher or decoding images during training.",
    "moss_repa_cache_size": "How many prepared image features to keep in system RAM for quicker access. 64 is the default. Features evicted from RAM are still read from disk, never recalculated during training. Larger values use more RAM; 0 reads from disk each time.",
    "moss_repa_cache_dir": "Leave blank to save REPA features in a moss_repa_features subfolder beside each dataset's latent cache. Or choose one shared cache folder with Browse. Preparation reuses valid files across epochs, restarts and runs, and rebuilds missing or outdated features automatically. This is separate from latent and caption caches.",
    "moss_dit_model": "Select the MOSS-Image diffusion model. Official BF16 and ComfyUI ConvRot INT8 checkpoints are supported. Base is the recommended starting point for training.",
    "moss_text_encoder": "Select the Qwen3.5-9B text encoder. BF16, ComfyUI ConvRot INT8 and W4A8 checkpoints are supported. W4A8 uses an experimental portable decoder and may be slow. The encoder turns captions into cached embeddings and is unloaded before training; it is different from Krea's Qwen3-VL encoder.",
    "moss_tokenizer_cache_dir": "Optional local tokenizer folder or download cache. Leave empty to download the small official MOSS tokenizer automatically.",
    "moss_text_encoder_cpu_streaming": "Use less graphics memory while reading captions by keeping encoder weights in system RAM and transferring them when needed. Slower, but useful on smaller GPUs. Does not change the training model.",
    "moss_max_token_length": "Maximum caption tokens, including MOSS's chat template. 4096 follows the official limit. A lower value can speed up long captions but may cut off details. Rebuild only the text cache after changing it.",
    "moss_convrot_int8": "Quantize a BF16 diffusion model while loading to save graphics memory. Leave off for a checkpoint already stored as ConvRot INT8; those files are detected automatically.",
    "moss_convrot_bwd_mode": "BF16 is the recommended gradient calculation for ConvRot weights. INT8 is an experimental speed option that approximates gradients and requires Triton on CUDA.",
    "moss_lora_target_preset": "All main-block projections trains attention and feed-forward layers. Attention only is a narrower experiment. Neither option trains the text encoder.",
    "moss_sample_flow_shift": "Controls the preview denoising schedule. 5 follows MOSS's official generation settings. This does not change the training timestep distribution.",
    "moss_sample_skip_layers": "Official MOSS previews skip block 22 in the negative guidance pass. Leave 22 for the standard behavior, or clear this field to disable skipping. It never skips a block during training.",
    "moss_pretraining_schedule": "Use the noise levels from MOSS's published pretraining recipe. Recommended as a starting point. Turn this off if you want to experiment with the general Timestep Sampling controls instead.",
    "moss_logit_mean": "Center of the MOSS pretraining noise distribution. The published value is -1.09. Changing it changes which noise levels are seen most often.",
    "moss_logit_std": "Spread of the MOSS pretraining noise distribution. The published value is 0.95; it must be positive.",
    "moss_timestep_cutoff": "Truncation point of the MOSS pretraining noise distribution before it is rescaled. The published value is 0.95; valid values are greater than zero and at most one.",
    "moss_timestep_uniform_ratio": "Fraction of steps that use a uniformly selected noise level alongside the main MOSS distribution. The published value 0.01 means about one in a hundred steps.",
    "moss_depth_anchor_weight": "0 turns depth guidance off. Try 0.01 for a first experiment, or 0.005 for gentler guidance. 0.01 passed a short MOSS GPU test; long-run quality is unverified. Depth compares the training image's estimated structure with the model's prediction. Enabling it adds VAE decoding, VRAM use and time; a smaller weight does not meaningfully reduce that cost.",
    "moss_depth_anchor_model": "Frozen Depth Anything helper used to estimate structure. Small is the lowest-memory starting point and downloads on first use.",
    "moss_depth_anchor_input_size": "Size used for depth estimation. 518 is the default; smaller multiples of 14 reduce memory and detail. This does not change your dataset training resolution.",
    "moss_depth_anchor_gradient_weight": "Extra emphasis on edges in the depth maps. This concerns spatial depth changes, not the optimizer's gradient strength.",
    "moss_depth_anchor_grad_checkpoint": "Save memory in the depth helper by recomputing intermediate values during backward. Usually keep enabled; disabling can be faster but uses more memory.",
    "moss_depth_anchor_vae_device": "Where depth-related VAE decoding runs. Training uses the main GPU; secondary uses another GPU; CPU is much slower. The depth helper and training model stay on the training GPU. A second GPU must have enough free memory for decoding and its gradients.",
    "moss_keep_depth_helpers_on_gpu": "Keep the VAE and depth helper resident between steps for speed. Off moves them back to system RAM after backward to free graphics memory. Enable only with spare VRAM.",
    "moss_weight_noise_sigma": "Optional random perturbation of adapter weights after each update. 0 disables it. Experimental regularization; it can also harm learned details.",
    "moss_weight_noise_mode": "Relative scales noise to the size of each adapter tensor. Absolute uses the entered strength directly.",
    "moss_weight_noise_bound_norm": "Keep each adapter tensor's magnitude unchanged after adding noise, limiting magnitude drift. Does not guarantee preserved image quality.",
}
