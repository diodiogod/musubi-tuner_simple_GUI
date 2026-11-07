"""MOSS settings shared by both interfaces."""
MODE = "MOSS-Image (Experimental)"
DEFAULTS = {
    "moss_dit_model": "", "moss_text_encoder": "", "moss_tokenizer_cache_dir": "",
    "moss_max_token_length": "4096", "moss_text_encoder_cpu_streaming": True,
    "moss_convrot_int8": False, "moss_convrot_bwd_mode": "bf16",
    "moss_lora_target_preset": "All main-block projections",
    "moss_sample_flow_shift": "5.0", "moss_sample_skip_layers": "22",
    "moss_pretraining_schedule": True, "moss_logit_mean": "-1.09", "moss_logit_std": "0.95",
    "moss_timestep_cutoff": "0.95", "moss_timestep_uniform_ratio": "0.01",
    "moss_depth_anchor_weight": "0.0",
    "moss_depth_anchor_model": "depth-anything/Depth-Anything-V2-Small-hf",
    "moss_depth_anchor_input_size": "518", "moss_depth_anchor_gradient_weight": "0.5",
    "moss_depth_anchor_grad_checkpoint": True, "moss_depth_anchor_vae_device": "training",
    "moss_keep_depth_helpers_on_gpu": False,
    "moss_weight_noise_sigma": "0.0", "moss_weight_noise_mode": "relative",
    "moss_weight_noise_bound_norm": False,
    "moss_repa_enabled": False, "moss_repa_weight": "0.5",
    "moss_repa_encoder": "facebook/dinov2-small", "moss_repa_block": "8",
    "moss_repa_image_size": "518", "moss_repa_device": "training", "moss_repa_cache_size": "64",
    "moss_repa_cache_dir": "",
}
CHOICES = {
    "moss_convrot_bwd_mode": ["bf16", "int8"],
    "moss_lora_target_preset": ["All main-block projections", "Attention only"],
    "moss_depth_anchor_vae_device": ["training", "secondary", "cpu"],
    "moss_weight_noise_mode": ["relative", "absolute"],
    "moss_repa_device": ["training", "cpu"],
}
PATHS = {"moss_dit_model", "moss_text_encoder", "moss_tokenizer_cache_dir", "moss_repa_cache_dir"}
REPA_LABELS = {
    "moss_repa_enabled": "Use REPA visual guidance (experimental)",
    "moss_repa_weight": "REPA strength", "moss_repa_encoder": "DINOv2 model or local folder",
    "moss_repa_block": "MOSS layer to guide", "moss_repa_image_size": "Visual teacher image size",
    "moss_repa_device": "Teacher compute device", "moss_repa_cache_size": "Remembered images (RAM)",
    "moss_repa_cache_dir": "REPA feature cache folder",
}
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
    "or image size between runs, continue from LoRA weights with a fresh optimizer."
)
HELP = {
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
