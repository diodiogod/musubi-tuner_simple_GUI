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
    assert command[command.index("--text_encoder") + 1].endswith("te.safetensors")
    assert command[command.index("--minimax_h3_preview_decode_min_free_gb") + 1] == "9.0"
    assert command[command.index("--depth_anchor_vae_device") + 1] == "training"
    assert command[command.index("--depth_anchor_every_n_steps") + 1] == "1"
    assert "--h3_guidance_distillation_protection" in command
    assert command[command.index("--h3_dynamic_sigma_every_n_steps") + 1] == "1"
    assert command[command.index("--h3_guidance_distillation_sigma_min") + 1] == "0.15"


def test_hybrid_assistant_command_forwards_helper_and_sparse_preservation(tmp_path):
    settings = _settings(tmp_path) | {
        "minimax_h3_training_assistant_enabled": True,
        "minimax_h3_dynamic_sigma_enabled": True,
        "minimax_h3_dynamic_sigma_every_n_steps": "10",
        "minimax_h3_base_preservation_enabled": True,
        "minimax_h3_training_assistant": "ostris/repo/helper.safetensors",
        "minimax_h3_base_preservation_loss_weight": "0.05",
        "minimax_h3_base_preservation_every_n_steps": "10",
    }

    (command,) = minimax_h3.build_commands(settings)

    assert "--h3_training_assistant_enabled" in command
    assert "--h3_base_preservation_enabled" in command
    assert "--h3_guidance_distillation_protection" in command
    assert command[command.index("--h3_dynamic_sigma_every_n_steps") + 1] == "10"
    assert command[command.index("--h3_training_assistant") + 1] == "ostris/repo/helper.safetensors"
    assert command[command.index("--h3_base_preservation_loss_weight") + 1] == "0.05"
    assert command[command.index("--h3_base_preservation_every_n_steps") + 1] == "10"


def test_base_preservation_falls_back_to_base_reference_without_assistant(tmp_path):
    settings = _settings(tmp_path) | {
        "minimax_h3_training_assistant_enabled": False,
        "minimax_h3_dynamic_sigma_enabled": True,
        "minimax_h3_base_preservation_enabled": True,
        "minimax_h3_base_preservation_reference": "Base + assistant",
        "minimax_h3_base_preservation_loss_weight": "0.05",
        "minimax_h3_base_preservation_every_n_steps": "10",
    }

    (command,) = minimax_h3.build_commands(settings)

    assert "--h3_training_assistant_enabled" not in command
    assert command[command.index("--h3_base_preservation_reference") + 1] == "base"


def test_cache_commands_use_image_only_tools_and_compact_te(tmp_path):
    settings = _settings(tmp_path)
    settings.update(recache_latents=True, recache_text=True)
    commands = minimax_h3.build_cache_commands(settings, "python")
    assert commands[0][1].endswith("minimax_h3_image_cache_latents.py")
    assert commands[0][commands[0].index("--vae_dtype") + 1] == "float32"
    assert commands[1][1].endswith("minimax_h3_image_cache_text_encoder_outputs.py")
    assert commands[1][commands[1].index("--text_encoder") + 1].endswith("te.safetensors")
    assert commands[1][commands[1].index("--text_encoder_load_mode") + 1] == "auto"
    assert commands[1][commands[1].index("--cache_dtype") + 1] == "bfloat16"


def test_combined_assistant_and_periodic_dynamic_sigma_still_cache_unconditional_text(tmp_path):
    settings = _settings(tmp_path) | {
        "recache_text": True,
        "minimax_h3_training_assistant_enabled": True,
        "minimax_h3_dynamic_sigma_enabled": True,
        "minimax_h3_dynamic_sigma_every_n_steps": "10",
        "minimax_h3_base_preservation_enabled": False,
    }

    command = minimax_h3.build_cache_commands(settings, "python")[0]

    assert "--cache_h3_unconditional" in command


def test_training_command_exposes_h3_regularization_and_samples(tmp_path):
    settings = _settings(tmp_path)
    settings.update(
        {
            "weight_noise_sigma": "unused",
            "krea2_weight_noise_sigma": "0.01",
            "krea2_weight_noise_mode": "relative",
            "krea2_weight_noise_bound_norm": True,
            "krea2_depth_anchor_weight": "0.1",
            "krea2_depth_anchor_input_size": "518",
            "dop_enabled": True,
            "dop_loss_weight": "0.2",
            "dop_trigger_word": "sks",
            "dop_class_word": "person",
            "sample_prompts": str(tmp_path / "samples.txt"),
            "sample_every_n_epochs": "1",
        }
    )
    (command,) = minimax_h3.build_commands(settings)
    assert command[command.index("--weight_noise_sigma") + 1] == "0.01"
    assert command[command.index("--depth_anchor_weight") + 1] == "0.1"
    assert command[command.index("--dop_loss_weight") + 1] == "0.2"
    assert command[command.index("--sample_prompts") + 1].endswith("samples.txt")


def test_official_multimodal_workflow_uses_isolated_joint_av_trainer(tmp_path):
    settings = _settings(tmp_path) | {
        "minimax_h3_training_workflow": "Video + audio · official multimodal",
        "minimax_h3_multimodal_task": "fl2va",
        "minimax_h3_video_vae": str(tmp_path / "video_vae.safetensors"),
        "minimax_h3_audio_vae": str(tmp_path / "audio_vae.safetensors"),
        "minimax_h3_video_only": False,
        "minimax_h3_audio_loss_weight": "0.5",
        "minimax_h3_dynamic_sigma_enabled": True,
    }
    (command,) = minimax_h3.build_commands(settings)

    assert command[6] == "src/musubi_tuner/minimax_h3_native_train_network.py"
    assert command[command.index("--task") + 1] == "fl2va"
    assert command[command.index("--audio_loss_weight") + 1] == "0.5"
    assert command[command.index("--text_encoder_blocks_to_swap") + 1] == "50"
    assert command[command.index("--timestep_sampling") + 1] == "uniform"
    assert command[command.index("--weighting_scheme") + 1] == "none"
    assert command[command.index("--discrete_flow_shift") + 1] == "1.0"
    assert "--h3_guidance_loss_scale" in command
    assert "--h3_guidance_loss_uncond_cache" in command
    assert "--h3_training_assistant" not in command


def test_official_multimodal_cache_sequence_includes_audio_and_guidance_probe(tmp_path):
    settings = _settings(tmp_path) | {
        "minimax_h3_training_workflow": "Video + audio · official multimodal",
        "minimax_h3_multimodal_task": "t2va",
        "minimax_h3_video_vae": str(tmp_path / "video_vae.safetensors"),
        "minimax_h3_audio_vae": str(tmp_path / "audio_vae.safetensors"),
        "minimax_h3_dynamic_sigma_enabled": True,
        "recache_latents": True,
        "recache_text": True,
    }
    latent, text = minimax_h3.build_cache_commands(settings, "python")

    assert latent[1].endswith("minimax_h3_native_cache_latents.py")
    assert latent[latent.index("--task") + 1] == "t2va"
    assert latent[latent.index("--audio_vae") + 1].endswith("audio_vae.safetensors")
    assert text[1].endswith("minimax_h3_native_cache_text_encoder_outputs.py")
    assert text[text.index("--text_encoder_blocks_to_swap") + 1] == "50"
    assert "--uncond_output" in text


def test_official_multimodal_audio_only_is_explicit(tmp_path):
    settings = _settings(tmp_path) | {
        "minimax_h3_training_workflow": "Video + audio · official multimodal",
        "minimax_h3_training_target": "Audio only (experimental)",
    }

    (command,) = minimax_h3.build_commands(settings)

    assert "--audio_only" in command
    assert "--video_only" not in command
