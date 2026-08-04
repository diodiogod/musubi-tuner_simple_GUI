from pathlib import Path

from modern_gui.prompt_preview import build_krea_preview, serialize_prompt
from modern_gui.sample_prompts import serialize_sample_prompt


def test_serialize_krea_preview_prompt():
    assert serialize_prompt({"prompt": "portrait", "width": 512, "seed": 42, "neg": "blur"}) == (
        "portrait --w 512 --d 42 --n blur"
    )


def test_serialize_training_prompt_preserves_mode_specific_flags():
    prompt = {
        "prompt": "motion",
        "width": "832",
        "height": "480",
        "steps": "20",
        "guidance": "5",
        "frames": "25",
        "flow_shift": "3",
        "cfg_scale": "1",
        "seed": "0",
        "neg": "blur",
        "image_path": "start.png",
    }

    assert serialize_sample_prompt(prompt, "Wan 2.2") == (
        "motion --w 832 --h 480 --s 20 --g 5 --f 25 --fs 3 "
        "--l 1 --d 0 --n blur --i start.png"
    )


def test_build_krea_preview_uses_turbo_and_batch_file(tmp_path: Path):
    models = {}
    for name in ("raw.safetensors", "turbo.safetensors", "vae.safetensors", "text.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        models[name] = str(path)
    command, save_path = build_krea_preview(
        {
            "training_mode": "Krea 2",
            "krea2_dit_model": models["raw.safetensors"],
            "krea2_turbo_dit": models["turbo.safetensors"],
            "vae_model": models["vae.safetensors"],
            "krea2_text_encoder": models["text.safetensors"],
            "output_dir": str(tmp_path),
            "output_name": "preview",
            "attention_mechanism": "sdpa",
        },
        [{"prompt": "one", "enabled": True}, {"prompt": "two", "enabled": True}],
    )

    assert "--turbo" in command
    assert "--from_file" in command
    assert save_path == tmp_path / "preview" / "sample_test"
    prompt_file = Path(command[command.index("--from_file") + 1])
    assert prompt_file.parent == save_path
    assert prompt_file.name.startswith("preview_prompts_")
    assert prompt_file.read_text(encoding="utf-8") == "one\ntwo"
    assert not list(save_path.glob(".preview-prompts-*.tmp"))

    second_command, _ = build_krea_preview(
        {
            "training_mode": "Krea 2",
            "krea2_dit_model": models["raw.safetensors"],
            "krea2_turbo_dit": models["turbo.safetensors"],
            "vae_model": models["vae.safetensors"],
            "krea2_text_encoder": models["text.safetensors"],
            "output_dir": str(tmp_path),
            "output_name": "preview",
            "attention_mechanism": "sdpa",
        },
        [{"prompt": "replacement", "enabled": True}, {"prompt": "batch", "enabled": True}],
    )
    second_prompt_file = Path(second_command[second_command.index("--from_file") + 1])
    assert second_prompt_file != prompt_file
    assert prompt_file.read_text(encoding="utf-8") == "one\ntwo"
    assert second_prompt_file.read_text(encoding="utf-8") == "replacement\nbatch"
