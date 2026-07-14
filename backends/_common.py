import os
from pathlib import Path


def normalize_path(p):
    return p.replace(os.sep, '/') if isinstance(p, str) and p else p


def add_arg(cmd, key, value, is_path=False):
    if value is None:
        return
    clean = str(value).strip()
    if clean in ("", "False"):
        return
    if clean in (True, "True"):
        cmd.append(key)
    else:
        cmd.extend([key, normalize_path(clean) if is_path else clean])


def build_output_dir(settings, suffix=""):
    output_dir = Path(settings["output_dir"]) / (settings["output_name"] + suffix)
    os.makedirs(output_dir, exist_ok=True)
    return str(output_dir), settings["output_name"] + suffix


def build_network_args(cmd, settings, default_lora_module):
    net_map = {"LoHa": "networks.loha", "LoKr": "networks.lokr"}
    net_type = settings.get("network_type", "LoRA")
    module = net_map.get(net_type, default_lora_module)
    add_arg(cmd, "--network_module", module)
    add_arg(cmd, "--network_dim", settings.get("network_dim_low"))
    add_arg(cmd, "--network_alpha", settings.get("network_alpha_low"))
    if net_type == "LoKr":
        factor = (settings.get("lokr_factor") or "").strip()
        if factor and factor != "-1":
            add_arg(cmd, "--network_args", f"factor={factor}")
    return net_type


def build_attention_arg(cmd, settings):
    attn = settings.get("attention_mechanism")
    if attn and attn != "none":
        cmd.append(f"--{attn}")


def build_sample_args(cmd, settings):
    sample_prompts = settings.get("sample_prompts", "")
    if not sample_prompts:
        return
    add_arg(cmd, "--sample_prompts", sample_prompts, is_path=True)
    # Only pass if > 0 — passing 0 causes ZeroDivisionError in the trainer (epoch % 0)
    n_epochs = str(settings.get("sample_every_n_epochs") or "").strip()
    if n_epochs and n_epochs != "0":
        add_arg(cmd, "--sample_every_n_epochs", n_epochs)
    n_steps = str(settings.get("sample_every_n_steps") or "").strip()
    if n_steps and n_steps != "0":
        add_arg(cmd, "--sample_every_n_steps", n_steps)
    if settings.get("sample_at_first"):
        cmd.append("--sample_at_first")


def build_common_train_args(cmd, settings):
    add_arg(cmd, "--optimizer_type", settings.get("optimizer_type"))
    add_arg(cmd, "--learning_rate", settings.get("learning_rate"))
    add_arg(cmd, "--max_grad_norm", settings.get("max_grad_norm"))
    add_arg(cmd, "--gradient_checkpointing", settings.get("gradient_checkpointing"))
    add_arg(cmd, "--gradient_accumulation_steps", settings.get("gradient_accumulation_steps"))
    add_arg(cmd, "--max_data_loader_n_workers", settings.get("max_data_loader_n_workers"))
    add_arg(cmd, "--persistent_data_loader_workers", settings.get("persistent_data_loader_workers"))
    add_arg(cmd, "--timestep_sampling", settings.get("timestep_sampling"))
    add_arg(cmd, "--num_timestep_buckets", settings.get("num_timestep_buckets"))
    add_arg(cmd, "--discrete_flow_shift", settings.get("discrete_flow_shift"))
    add_arg(cmd, "--preserve_distribution_shape", settings.get("preserve_distribution_shape"))
    if settings.get("compile"):
        cmd.append("--compile")
        add_arg(cmd, "--compile_backend", settings.get("compile_backend"))
        add_arg(cmd, "--compile_mode", settings.get("compile_mode"))
        add_arg(cmd, "--compile_dynamic", settings.get("compile_dynamic"))
        add_arg(cmd, "--compile_fullgraph", settings.get("compile_fullgraph"))
        add_arg(cmd, "--compile_cache_size_limit", settings.get("compile_cache_size_limit"))
    opt_args = (settings.get("optimizer_args") or "").strip()
    if opt_args:
        # Split on comma/semicolon/whitespace but NOT inside parentheses (e.g. betas=(0.9,0.99))
        import re as _re
        tokens = [t.strip() for t in _re.split(r'[,;\s]+(?![^(]*\))', opt_args) if t.strip()]
        for token in tokens:
            cmd.extend(["--optimizer_args", token])

    lr = settings.get("lr_scheduler")
    if lr and lr != "constant":
        add_arg(cmd, "--lr_scheduler", lr)
        if lr == "constant_with_warmup":
            add_arg(cmd, "--lr_warmup_steps", settings.get("lr_warmup_steps"))
        if lr == "cosine_with_restarts":
            add_arg(cmd, "--lr_scheduler_num_cycles", settings.get("lr_scheduler_num_cycles"))
        add_arg(cmd, "--lr_scheduler_power", settings.get("lr_scheduler_power"))
        add_arg(cmd, "--lr_scheduler_min_lr_ratio", settings.get("lr_scheduler_min_lr_ratio"))

    add_arg(cmd, "--max_train_steps", settings.get("max_train_steps"))
    add_arg(cmd, "--max_train_epochs", settings.get("max_train_epochs"))
    add_arg(cmd, "--save_every_n_epochs", settings.get("save_every_n_epochs"))
    add_arg(cmd, "--save_every_n_steps", settings.get("save_every_n_steps"))
    add_arg(cmd, "--seed", settings.get("seed"))
    add_arg(cmd, "--save_state", settings.get("save_state"))
    add_arg(cmd, "--resume", settings.get("resume_path"), is_path=True)
    add_arg(cmd, "--resume_exact_position", settings.get("resume_exact_position"))
    add_arg(cmd, "--network_weights", settings.get("network_weights"), is_path=True)
    add_arg(cmd, "--training_comment", settings.get("training_comment"))

    log = settings.get("log_with")
    if log and log != "none":
        add_arg(cmd, "--log_with", log)
        add_arg(cmd, "--logging_dir", settings.get("logging_dir"), is_path=True)
        add_arg(cmd, "--log_prefix", settings.get("log_prefix"))
