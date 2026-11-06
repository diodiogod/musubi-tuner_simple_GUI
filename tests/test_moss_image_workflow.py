"""Cross-component checks for a MOSS recipe: GUI -> commands -> runtime parsers."""
import json
from pathlib import Path

import torch
import pytest
from safetensors.torch import save_file

from backends.moss_settings import DEFAULTS, MODE
from modern_gui.commands import build_command_plan
from modern_gui.face_stages import prepare_face_stage
from modern_gui.prompt_preview import build_prompt_preview
from modern_gui.recovery import prepare_face_refinement
from modern_gui.sample_prompts import serialize_sample_prompt
from modern_gui.settings import settings_schema
from modern_gui.stages import apply_depth_memory_override
from modern_gui.validation import validate_training_settings
from musubi_tuner.face_refinement.lora_validation import validate_face_lora
from musubi_tuner.moss_image_cache_latents import setup_parser as latent_parser
from musubi_tuner.moss_image_cache_text_encoder_outputs import setup_parser as text_parser
from musubi_tuner.moss_image_face_refinement import build_parser as face_parser
from musubi_tuner.moss_image_generate import setup_parser as preview_parser
from musubi_tuner.moss_image_train_network import moss_setup_parser, setup_parser_common, MossImageNetworkTrainer


def recipe(tmp_path):
    settings = json.loads(Path("Base_SETTINGS.json").read_text(encoding="utf-8"))
    settings.update(DEFAULTS)
    for name in ("dit", "vae", "text"):
        (tmp_path / name).touch()
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[general]\n", encoding="utf-8")
    settings.update(training_mode=MODE, moss_dit_model=str(tmp_path / "dit"), vae_model=str(tmp_path / "vae"),
                    moss_text_encoder=str(tmp_path / "text"), dataset_config=str(dataset),
                    output_dir=str(tmp_path), output_name="moss", attention_mechanism="sdpa",
                    mixed_precision="bf16", recache_text=True, recache_latents=True,
                    dop_enabled=True, dop_trigger_word="person_token", dop_class_word="a person", dop_loss_weight="1.0")
    return settings


def test_gui_recipe_reaches_all_runtime_parsers(tmp_path):
    settings = recipe(tmp_path)
    plan = build_command_plan(settings, preview=True)
    latent_cmd, text_cmd = plan["cache"]
    latent_args = latent_parser().parse_args(latent_cmd[2:])
    text_args = text_parser().parse_args(text_cmd[2:])
    command = plan["train"][0]
    script = command.index("src/musubi_tuner/moss_image_train_network.py")
    args = moss_setup_parser(setup_parser_common()).parse_args(command[script + 1:])
    MossImageNetworkTrainer().handle_model_specific_args(args)
    assert latent_args.vae == settings["vae_model"]
    assert text_args.dop_trigger_word == args.dop_trigger_word == "person_token"
    assert args.moss_pretraining_schedule and args.moss_logit_mean == -1.09
    assert args.network_module == "networks.lora_moss_image"
    fields = {field["key"]: field for section in settings_schema(settings)["sections"] for field in section["fields"]}
    assert fields["moss_dit_model"]["modes"] == [MODE] and fields["moss_dit_model"]["help"]


def test_dashboard_has_a_choice_for_every_supported_mode():
    from html.parser import HTMLParser
    from modern_gui.settings import MODES

    class ModelTiles(HTMLParser):
        def __init__(self):
            super().__init__()
            self.modes = set()

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "button" and "model-tile" in attrs.get("class", "").split():
                self.modes.add(attrs.get("data-mode"))

    tiles = ModelTiles()
    tiles.feed(Path("modern_gui/static/index.html").read_text(encoding="utf-8"))
    assert set(MODES) <= tiles.modes


def test_moss_validation_checks_face_encoder_depth_and_preview_parameters(tmp_path):
    settings = recipe(tmp_path)
    settings.update(recache_text=False, sample_prompts_data=[], sample_prompts="", moss_text_encoder="",
                    use_staged_training=True,
                    staged_training_config=[{"enabled": True, "type": "face_refinement"}],
                    moss_depth_anchor_weight="0.01", moss_depth_anchor_input_size="500",
                    moss_sample_flow_shift="nan", moss_sample_skip_layers="42")
    errors = {item["key"] for item in validate_training_settings(settings)["errors"]}
    assert {"moss_text_encoder", "moss_depth_anchor_input_size", "moss_sample_flow_shift", "moss_sample_skip_layers"} <= errors


def test_moss_preview_matches_scheduled_prompt_controls(tmp_path):
    settings = recipe(tmp_path)
    settings["preview_use_lora"] = False
    prompt = {"prompt": "photo of a person", "guidance": "3.2", "flow_shift": "5.0", "steps": "20",
              "width": "512", "height": "512", "neg": "blur"}
    command, _ = build_prompt_preview(settings, [prompt])
    args = preview_parser().parse_args(command[2:])
    assert args.cfg_scale == 3.2 and args.negative_prompt == "blur" and args.flow_shift == 5.0
    line = serialize_sample_prompt(prompt, MODE)
    assert "--l 3.2" in line and "--fs 5.0" in line and "--g " not in line


def test_moss_preview_preserves_zero_seed_and_rejects_invalid_shift(tmp_path):
    settings = recipe(tmp_path)
    settings["preview_use_lora"] = False
    prompt = {"prompt": "a portrait", "seed": 0, "guidance": 0, "flow_shift": 5}
    command, _ = build_prompt_preview(settings, [prompt])
    args = preview_parser().parse_args(command[2:])
    assert args.seed == 0 and args.cfg_scale == 0
    for bad_shift in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="shift"):
            build_prompt_preview(settings, [dict(prompt, flow_shift=bad_shift)])


def test_moss_face_stage_is_validated_and_built_with_moss_backend(tmp_path):
    settings = recipe(tmp_path)
    lora = tmp_path / "subject.safetensors"
    save_file({"lora_unet_layers_0_proj_q.lora_down.weight": torch.ones(2, 4),
               "lora_unet_layers_0_proj_q.lora_up.weight": torch.ones(4, 2)}, str(lora))
    assert validate_face_lora(lora, MODE)["attention_modules"] == 1
    settings["face_refinement_config"] = {
        "resolution": 512, "blocks_to_swap": 30, "denoise_steps": 10, "draft_k": 1,
        "reference_dir": str(tmp_path), "face_model_dir": str(tmp_path), "prompts": ["portrait"],
        "preflight_report": {"scored_images": [{"path": str(tmp_path / "face.png")} ]},
    }
    _, command, _ = prepare_face_stage(settings, {"type": "face_refinement", "steps": "5", "label": "face"}, 0, lora)
    assert command[1].endswith("moss_image_face_refinement.py")
    args = face_parser().parse_args(command[2:])
    assert Path(args.dit) == Path(settings["moss_dit_model"]) and args.train_steps == 5
    apply_depth_memory_override(settings, {"depth_helpers_mode": "keep on GPU"})
    assert settings["moss_keep_depth_helpers_on_gpu"] is True


def test_moss_face_refinement_can_start_from_job_history(tmp_path):
    run = tmp_path / "subject"
    run.mkdir()
    lora = run / "subject.safetensors"
    save_file({"lora_unet_layers_0_proj_q.lora_down.weight": torch.ones(2, 4),
               "lora_unet_layers_0_proj_q.lora_up.weight": torch.ones(4, 2)}, str(lora))
    settings = prepare_face_refinement({"settings_snapshot": {
        "training_mode": MODE, "output_name": "subject", "output_dir": str(tmp_path),
        "max_train_epochs": ""}})
    face = settings["face_refinement_config"]
    assert face["input_lora"] == str(lora.resolve())
    assert face["cfg_scale"] == 3.0 and face["blocks_to_swap"] == 30
    assert settings["staged_training_config"][0]["type"] == "face_refinement"
