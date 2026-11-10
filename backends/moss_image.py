"""Shared classic/modern GUI command construction for MOSS-Image."""
from backends._common import (
    add_arg, apply_cache_policy, build_output_dir, build_network_args,
    build_attention_arg, build_common_train_args, build_sample_args,
    build_dop_cache_args, build_dop_train_args,
)


def _text_options(cmd, settings):
    add_arg(cmd, "--text_encoder", settings.get("moss_text_encoder"), is_path=True)
    add_arg(cmd, "--tokenizer_cache_dir", settings.get("moss_tokenizer_cache_dir"), is_path=True)
    add_arg(cmd, "--moss_max_token_length", settings.get("moss_max_token_length", 4096))
    add_arg(cmd, "--text_encoder_cpu_streaming", settings.get("moss_text_encoder_cpu_streaming", True))


def build_commands(settings):
    settings = training_options(settings)
    cmd = ["accelerate", "launch", "--num_processes", "1", "--num_cpu_threads_per_process", "1",
           "src/musubi_tuner/moss_image_train_network.py"]
    add_arg(cmd, "--mixed_precision", settings.get("mixed_precision", "bf16"))
    add_arg(cmd, "--dit", settings.get("moss_dit_model"), is_path=True)
    add_arg(cmd, "--vae", settings.get("vae_model"), is_path=True)
    add_arg(cmd, "--dataset_config", settings.get("dataset_config"), is_path=True)
    _text_options(cmd, settings)
    if settings.get("moss_ema_enabled"):
        for key in ("moss_ema_enabled", "moss_ema_decay", "moss_ema_device"):
            add_arg(cmd, "--" + key, settings.get(key))
    add_arg(cmd, "--convrot_int8", settings.get("moss_convrot_int8", False))
    add_arg(cmd, "--convrot_bwd_mode", settings.get("moss_convrot_bwd_mode", "bf16"))
    add_arg(cmd, "--blocks_to_swap", settings.get("blocks_to_swap"))
    for key in ("moss_unpadded_batches", "moss_text_pad_multiple", "moss_compile_checkpoint", "moss_sdpa_backend"):
        add_arg(cmd, "--" + key, settings.get(key))
    for name in ("weight_noise_sigma", "weight_noise_mode", "weight_noise_bound_norm",
                 "depth_anchor_weight", "depth_anchor_model", "depth_anchor_input_size",
                 "depth_anchor_gradient_weight", "depth_anchor_vae_device", "keep_depth_helpers_on_gpu"):
        add_arg(cmd, "--" + name, settings.get("moss_" + name))
    if not settings.get("moss_depth_anchor_grad_checkpoint", True):
        cmd.append("--no-depth_anchor_grad_checkpoint")
    if settings.get("moss_repa_enabled", False):
        from backends.moss_settings import DEFAULTS
        cmd.append("--moss_repa_enabled")
        for key in DEFAULTS:
            if key.startswith("moss_repa_") and key != "moss_repa_enabled":
                add_arg(cmd, "--" + key, settings.get(key, DEFAULTS[key]))
    add_arg(cmd, "--moss_sample_flow_shift", settings.get("moss_sample_flow_shift", 5.0))
    for key in ("moss_sample_sampler", "moss_sample_scheduler", "moss_sample_lora", "moss_sample_lora_multiplier"):
        add_arg(cmd, "--" + key, settings.get(key), is_path=key == "moss_sample_lora")
    if settings.get("moss_pretraining_schedule", True):
        cmd.append("--moss_pretraining_schedule")
        add_arg(cmd, "--moss_clean_end_share", settings.get("moss_clean_end_share"))
        for key in ("moss_logit_mean", "moss_logit_std", "moss_timestep_cutoff", "moss_timestep_uniform_ratio"):
            add_arg(cmd, "--" + key, settings.get(key))
    elif settings["moss_timestep_schedule"] == "Sigmoid + High (AI Toolkit)":
        cmd.append("--moss_sigmoid_high")
    # Preserve an explicit empty value: it disables the official skip-layer guidance.
    cmd.extend(["--moss_sample_skip_layers", str(settings.get("moss_sample_skip_layers", "22"))])
    target_args = []
    if settings.get("moss_fast_lora"):
        target_args.append("moss_lowp_lora=True")
    if settings.get("moss_lora_target_preset") == "Attention only":
        target_args.append("exclude_patterns=['.*[.]ffn_up_gate','.*[.]ffn_down']")
    build_network_args(cmd, settings, "networks.lora_moss_image", target_args)
    build_attention_arg(cmd, settings)
    build_sample_args(cmd, settings)
    build_dop_train_args(cmd, settings)
    build_common_train_args(cmd, settings)
    output_dir, output_name = build_output_dir(settings)
    add_arg(cmd, "--output_dir", output_dir, is_path=True)
    add_arg(cmd, "--output_name", output_name)
    return [cmd]


def training_options(settings):
    """Validate new opt-ins on a copy; never rewrite the saved recipe."""
    import math
    from .numeric_settings import normalize_numbers, number_value
    settings = normalize_numbers(settings)
    from .moss_settings import DEFAULTS, with_timestep_schedule
    from .regularization_switches import effective_settings
    result = with_timestep_schedule(effective_settings(settings))
    if str(settings.get("moss_clean_end_share") or "").strip():
        from musubi_tuner.moss_image.timesteps import clean_end_policy
        clean_end_policy(settings["moss_clean_end_share"], 1)
    multiple = float(settings.get("moss_text_pad_multiple", 0) or 0)
    if not math.isfinite(multiple) or multiple != int(multiple) or not 0 <= multiple <= 512:
        raise ValueError("MOSS caption padding must be a whole number from 0 to 512. Try 16, or 0 to disable.")
    if settings.get("moss_unpadded_batches") and multiple:
        raise ValueError("Unpadded MOSS image passes require Caption padding = 0.")
    if settings.get("moss_fast_lora") and str(settings.get("network_type", "LoRA")).lower() != "lora":
        raise ValueError("Fast MOSS LoRA compute supports plain LoRA. Turn it off to use LoKr/LoHa.")
    if settings.get("moss_fast_lora") and (settings.get("full_bf16") or settings.get("full_fp16")):
        raise ValueError("Fast MOSS LoRA keeps FP32 master weights. Turn off Full BF16/FP16; mixed precision BF16 is supported.")
    if settings.get("compile") and int(settings.get("blocks_to_swap") or 0):
        raise ValueError("MOSS Compile requires Blocks to swap = 0. Turn Compile off to use swapping.")
    if str(settings.get("optimizer_type", "")).lower() == "muon":
        from ._common import split_optimizer_args
        import ast
        from musubi_tuner.optimizers.muon import validate_muon
        tokens = split_optimizer_args(settings.get("optimizer_args"))
        try:
            options = dict(token.split("=", 1) for token in tokens)
            if len(options) != len(tokens):
                raise ValueError("Duplicate Muon option.")
            options = {key: value if key == "adjust_lr_fn" and value in {"original", "match_rms_adamw", "spectral_unclamped"}
                       else ast.literal_eval(value) for key, value in options.items()}
            validate_muon(float(settings.get("learning_rate") or 0), **options)
        except (SyntaxError, TypeError, ValueError) as exc:
            raise ValueError(f"Check Muon Optimizer Args: {exc}. Use quoted strings, e.g. adjust_lr_fn=\"match_rms_adamw\".") from exc
    if str(settings.get("optimizer_type", "")).lower() == "automagic4":
        if settings.get("full_bf16") or settings.get("full_fp16"):
            raise ValueError("Automagic4 needs FP32 adapter weights; mixed precision BF16 is supported.")
        if settings.get("lr_scheduler", "constant") != "constant" or float(settings.get("lr_warmup_steps") or 0):
            raise ValueError("Automagic4 controls its own rate. Select constant scheduling and zero warmup.")
    resume = str(settings.get("resume_path") or "").strip()
    if resume:
        import json
        from pathlib import Path
        folder = Path(resume)
        if folder.is_dir():
            from musubi_tuner.optimizers.config import validate_optimizer_resume
            validate_optimizer_resume(folder, settings.get("optimizer_type", "adamw8bit"))
            marker = folder / "moss_ema.json"
            if marker.is_file() != bool(settings.get("moss_ema_enabled")):
                raise ValueError("Keep EMA on/off as in the saved state, or continue from LoRA weights with a fresh optimizer.")
            if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("decay") != float(settings.get("moss_ema_decay", DEFAULTS["moss_ema_decay"])):
                raise ValueError("Full-state resume must retain the saved EMA smoothing value.")
    if settings.get("moss_ema_enabled"):
        value = float(settings.get("moss_ema_decay", DEFAULTS["moss_ema_decay"]))
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("MOSS EMA smoothing must be at least 0 and less than 1. Try 0.98.")
        result.setdefault("moss_ema_decay", DEFAULTS["moss_ema_decay"])
        result.setdefault("moss_ema_device", DEFAULTS["moss_ema_device"])
        if result["moss_ema_device"] not in {"training", "cpu"}:
            raise ValueError("Choose training or cpu for MOSS EMA storage.")
    if str(settings.get("optimizer_type", "")).lower() == "automagic3":
        if settings.get("full_bf16") or settings.get("full_fp16"):
            raise ValueError("Automagic3 requires FP32 adapter weights. Turn off Full BF16/FP16; mixed precision BF16 is supported.")
        from ._common import split_optimizer_args
        tokens = split_optimizer_args(settings.get("optimizer_args"))
        names = [token.split("=", 1)[0] for token in tokens]
        allowed = {"beta2", "eps", "clip_threshold", "weight_decay", "polarity_history", "fused", "min_lr", "max_lr"}
        if set(names) - allowed:
            raise ValueError("Unsupported Automagic3 Optimizer Args: " + ", ".join(sorted(set(names) - allowed)) + ". It uses beta2, not AdamW betas.")
        if len(names) != len(set(names)):
            raise ValueError("Optimizer Args contains a duplicate option.")
        if any(name in names for name in ("min_lr", "max_lr")):
            raise ValueError("Set Automagic3 minimum/maximum rates in their dedicated fields, not Optimizer Args.")
        if any(token.replace(" ", "").lower() == "fused=true" for token in tokens):
            raise ValueError("Automagic3 needs fused=False in Musubi.")
        if settings.get("lr_scheduler", "constant") != "constant" or float(settings.get("lr_warmup_steps") or 0):
            raise ValueError("Automagic3 controls its own rate. Select the constant scheduler and zero warmup.")
        lo = number_value(settings, "moss_automagic3_min_lr", DEFAULTS["moss_automagic3_min_lr"])
        hi = number_value(settings, "moss_automagic3_max_lr", DEFAULTS["moss_automagic3_max_lr"])
        start = number_value(settings, "learning_rate", 0)
        if not all(math.isfinite(v) and v > 0 for v in (lo, hi, start)) or not lo <= start <= hi:
            raise ValueError("Automagic3 rates must be finite and positive, with minimum <= Learning Rate <= maximum.")
        import ast
        from musubi_tuner.optimizers.config import validate_automagic3
        try:
            options = {name: ast.literal_eval(token.split("=", 1)[1]) for name, token in zip(names, tokens)}
        except (SyntaxError, TypeError, IndexError, ValueError) as exc:
            raise ValueError("Check Automagic3 Optimizer Args; use name=value with numeric values or True/False.") from exc
        validate_automagic3(start, min_lr=lo, max_lr=hi, **options)
        tokens += [f"min_lr={lo}", f"max_lr={hi}"]
        if "weight_decay" not in names:
            tokens.append("weight_decay=0.00001")
        result["optimizer_args"] = " ".join(tokens)
    return result


def build_cache_commands(settings, python_executable):
    commands = []
    if settings.get("recache_latents"):
        commands.append([python_executable, "src/musubi_tuner/moss_image_cache_latents.py",
                         "--dataset_config", settings["dataset_config"], "--vae", settings["vae_model"]])
    if settings.get("recache_text"):
        cmd = [python_executable, "src/musubi_tuner/moss_image_cache_text_encoder_outputs.py",
               "--dataset_config", settings["dataset_config"]]
        _text_options(cmd, settings)
        build_dop_cache_args(cmd, settings)
        commands.append(cmd)
    return apply_cache_policy(commands, settings)
