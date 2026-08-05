import os
from pathlib import Path

from modern_gui.commands import build_command_plan
from modern_gui.sample_prompts import planned_sample_prompt_path


def test_krea_plan_uses_existing_backend_adapters():
    settings = {
        "training_mode": "Krea 2",
        "dataset_config": "dataset.toml",
        "krea2_dit_model": "raw.safetensors",
        "vae_model": "vae.safetensors",
        "krea2_text_encoder": "text",
        "output_dir": "output",
        "output_name": "test",
        "recache_latents": True,
        "recache_text": True,
    }

    plan = build_command_plan(settings)

    assert any("krea2_cache_latents.py" in item for item in plan["cache"][0])
    assert any("krea2_train_network.py" in item for item in plan["train"][0])
    assert "--num_processes" in plan["train"][0]


def test_minimax_h3_plan_uses_direct_pruned_backend(tmp_path):
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "dataset_config": "dataset.toml",
        "minimax_h3_dit_model": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "minimax_h3_text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "minimax_h3_tokenizer": "Qwen/Qwen3-VL-32B-Instruct",
        "vae_model": "vae.safetensors",
        "output_dir": str(tmp_path),
        "output_name": "h3",
        "network_type": "LoRA",
        "network_dim_low": "16",
        "network_alpha_low": "16",
        "blocks_to_swap": "30",
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
        "attention_mechanism": "sdpa",
        "recache_text": True,
        "minimax_h3_guidance_distillation_protection": True,
        "minimax_h3_guidance_distillation_scale": "3.0",
    }

    plan = build_command_plan(settings, preview=True)

    assert any("minimax_h3_image_cache_text_encoder_outputs.py" in item for item in plan["cache"][0])
    command = plan["train"][0]
    assert "src/musubi_tuner/minimax_h3_image_train_network.py" in command
    assert "--block_swap_h2d_only" in command
    assert "--h3_guidance_distillation_protection" in command
    assert command[command.index("--h3_guidance_distillation_scale") + 1] == "3.0"
    assert "--cache_h3_unconditional" in plan["cache"][0]
    assert "--fp8_base" not in command


def test_training_uses_accelerate_from_gui_python_environment(tmp_path):
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "dataset_config": "dataset.toml",
        "minimax_h3_dit_model": "dit.safetensors",
        "output_dir": str(tmp_path),
        "output_name": "h3",
        "network_type": "LoRA",
        "network_dim_low": "16",
        "network_alpha_low": "16",
        "blocks_to_swap": "30",
        "mixed_precision": "bf16",
    }

    launcher = Path(build_command_plan(settings, preview=True)["train"][0][0])

    assert launcher.name == ("accelerate.exe" if os.name == "nt" else "accelerate")
    assert launcher.is_file()


def test_preview_does_not_create_output_directory(tmp_path):
    settings = {
        "training_mode": "Krea 2",
        "dataset_config": "dataset.toml",
        "krea2_dit_model": "raw.safetensors",
        "vae_model": "vae.safetensors",
        "krea2_text_encoder": "text",
        "output_dir": str(tmp_path),
        "output_name": "must-not-exist",
    }

    build_command_plan(settings, preview=True)

    assert not (tmp_path / "must-not-exist").exists()


def test_preview_includes_visual_prompt_path_without_writing_it(tmp_path):
    settings = {
        "training_mode": "Krea 2",
        "dataset_config": "dataset.toml",
        "krea2_dit_model": "raw.safetensors",
        "vae_model": "vae.safetensors",
        "krea2_text_encoder": "text",
        "output_dir": str(tmp_path),
        "output_name": "prompted",
        "sample_every_n_epochs": "1",
        "sample_prompts_data": [
            {
                "enabled": True,
                "prompt": "portrait",
                "width": 1024,
                "height": 1024,
                "guidance": 5.5,
            }
        ],
    }

    plan = build_command_plan(settings, preview=True)
    prompt_path = planned_sample_prompt_path(settings)

    assert "--sample_prompts" in plan["train"][0]
    assert str(prompt_path).replace("\\", "/") in plan["train"][0]
    assert not prompt_path.exists()


def test_real_plan_writes_enabled_visual_prompts_for_backend(tmp_path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]", encoding="utf-8")
    settings = {
        "training_mode": "Krea 2",
        "dataset_config": str(dataset),
        "krea2_dit_model": "raw.safetensors",
        "vae_model": "vae.safetensors",
        "krea2_text_encoder": "text",
        "output_dir": str(tmp_path),
        "output_name": "prompted",
        "sample_every_n_steps": "20",
        "sample_prompts_data": [
            {"enabled": False, "prompt": "skip me"},
            {
                "enabled": True,
                "prompt": "portrait",
                "width": 1024,
                "seed": 0,
                "guidance": 5.5,
                "neg": "blur",
            },
        ],
    }

    plan = build_command_plan(settings)
    prompt_path = planned_sample_prompt_path(settings)

    assert prompt_path.read_text(encoding="utf-8") == (
        "portrait --w 1024 --l 5.5 --d 0 --n blur"
    )
    assert "--sample_every_n_steps" in plan["train"][0]


def test_starting_point_mode_clears_inactive_hidden_paths(tmp_path):
    state = tmp_path / "old-state"
    state.mkdir()
    weights = tmp_path / "old.safetensors"
    weights.write_bytes(b"lora")
    settings = {
        "training_mode": "Krea 2",
        "dataset_config": "dataset.toml",
        "krea2_dit_model": "raw.safetensors",
        "vae_model": "vae.safetensors",
        "krea2_text_encoder": "text",
        "output_dir": str(tmp_path),
        "output_name": "fresh",
        "starting_point_mode": "new",
        "resume_path": str(state),
        "network_weights": str(weights),
        "resume_exact_position": True,
        "recovery_mode": True,
    }

    command = build_command_plan(settings, preview=True)["train"][0]

    assert "--resume" not in command
    assert "--network_weights" not in command


def test_wan_i2v_cache_uses_filtered_execution_copy_without_touching_toml(tmp_path):
    dataset = tmp_path / "dataset.toml"
    original = (
        '[[datasets]]\n'
        'image_directory = "frames"\n'
        'video_directory = "clips"\n'
        'resolution = [832, 480]\n'
    )
    dataset.write_text(original, encoding="utf-8")
    settings = {
        "training_mode": "Wan 2.2",
        "dataset_config": str(dataset),
        "output_dir": str(tmp_path),
        "output_name": "i2v",
        "vae_model": "vae.safetensors",
        "t5_model": "t5.safetensors",
        "clip_model": "clip.safetensors",
        "dit_low_noise": "low.safetensors",
        "train_low_noise": True,
        "is_i2v": True,
        "recache_latents": True,
        "recache_text": True,
    }

    plan = build_command_plan(settings)
    cache_paths = [
        Path(command[command.index("--dataset_config") + 1])
        for command in plan["cache"]
    ]

    assert len(cache_paths) == 2
    assert cache_paths[0] == cache_paths[1]
    assert cache_paths[0] != dataset
    assert "image_directory" not in cache_paths[0].read_text(encoding="utf-8")
    assert 'video_directory = "clips"' in cache_paths[0].read_text(encoding="utf-8")
    assert dataset.read_text(encoding="utf-8") == original
    cache_paths[0].unlink()


def test_wan_i2v_command_preview_is_read_only(tmp_path):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text('[[datasets]]\nimage_directory = "frames"\n', encoding="utf-8")
    settings = {
        "training_mode": "Wan 2.2",
        "dataset_config": str(dataset),
        "output_dir": str(tmp_path),
        "output_name": "i2v-preview",
        "vae_model": "vae.safetensors",
        "t5_model": "t5.safetensors",
        "clip_model": "clip.safetensors",
        "dit_low_noise": "low.safetensors",
        "train_low_noise": True,
        "is_i2v": True,
        "recache_latents": True,
    }

    before = set(tmp_path.iterdir())
    plan = build_command_plan(settings, preview=True)

    assert plan["cache"][0][plan["cache"][0].index("--dataset_config") + 1] == str(dataset)
    assert set(tmp_path.iterdir()) == before
