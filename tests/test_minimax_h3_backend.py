from backends import minimax_h3


def _settings(tmp_path):
    return {
        "network_type": "LoRA",
        "mixed_precision": "bf16",
        "minimax_h3_dit_model": str(tmp_path / "dit.safetensors"),
        "minimax_h3_text_encoder": str(tmp_path / "te.safetensors"),
        "minimax_h3_tokenizer": "Qwen/Qwen3-VL-32B-Instruct",
        "vae_model": str(tmp_path / "vae.safetensors"),
        "dataset_config": str(tmp_path / "dataset.toml"),
        "output_dir": str(tmp_path),
        "output_name": "h3_test",
        "network_dim_low": "16",
        "network_alpha_low": "16",
        "blocks_to_swap": "30",
        "block_swap_ring_size": "2",
        "gradient_checkpointing": True,
        "timestep_sampling": "krea2_shift",
        "attention_mechanism": "sdpa",
        "minimax_h3_convrot_bwd_mode": "bf16",
        "_preview_only": True,
    }


def test_training_command_enforces_direct_int8_safe_path(tmp_path):
    (command,) = minimax_h3.build_commands(_settings(tmp_path))
    assert command[6] == "src/musubi_tuner/minimax_h3_image_train_network.py"
    assert "--block_swap_h2d_only" in command
    assert "--gradient_checkpointing" in command
    assert command[command.index("--network_module") + 1] == "networks.lora_minimax_h3"
    assert "--fp8_base" not in command


def test_cache_commands_use_image_only_tools_and_compact_te(tmp_path):
    settings = _settings(tmp_path)
    settings.update(recache_latents=True, recache_text=True)
    commands = minimax_h3.build_cache_commands(settings, "python")
    assert commands[0][1].endswith("minimax_h3_image_cache_latents.py")
    assert commands[1][1].endswith("minimax_h3_image_cache_text_encoder_outputs.py")
    assert commands[1][commands[1].index("--text_encoder") + 1].endswith("te.safetensors")
