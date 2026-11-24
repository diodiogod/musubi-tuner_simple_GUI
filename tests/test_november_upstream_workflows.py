import json
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from backends.lora_ema import plan_ema
from backends.image_masks import MODES as MASK_MODES
from backends.minimax_h3_picture import MODE, MODEL_FIELDS
from modern_gui.commands import build_command_plan
from modern_gui.prompt_preview import build_prompt_preview
from modern_gui.sample_prompts import prepare_sample_prompt_settings
from modern_gui.settings import MODES, settings_schema
from musubi_tuner import cache_latents, cache_text_encoder_outputs
from musubi_tuner.minimax_h3.image_sampling import H3_PICTURE_SIGMAS, sample_image_latent
from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model
from musubi_tuner.minimax_h3.video_vae import ViT3DDecoder
from musubi_tuner.minimax_h3_image_cache_latents import setup_parser as latent_parser
from musubi_tuner.minimax_h3_image_cache_text_encoder_outputs import setup_parser as text_parser
from musubi_tuner.minimax_h3_image_generate import setup_parser as preview_parser
from musubi_tuner.minimax_h3_image_train_network import MiniMaxH3PictureNetworkTrainer, minimax_h3_image_setup_parser
from musubi_tuner.dataset.image_loss_masks import mean_image_loss, prepare_masks
from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.run_control import RunControl
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer


def test_h3_image_recipe_routes_to_picture_only_parsers(tmp_path):
    weights = {}
    for key in MODEL_FIELDS:
        file = tmp_path / f"{key}.safetensors"
        save_file({"dummy": torch.zeros(1)}, str(file))
        weights[key] = str(file)
    settings = {
        **weights, "training_mode": MODE, "dataset_config": str(tmp_path / "dataset.toml"),
        "output_dir": str(tmp_path), "output_name": "picture", "network_dim_low": 4, "network_alpha_low": 4,
        "recache_latents": True, "recache_text": True, "blocks_to_swap": 4, "network_type": "LoRA",
        "minimax_h3_dynamic_sigma_enabled": True, "minimax_h3_training_workflow": "Video + images",
    }
    prompts = [{"prompt": "A mug", "width": 512, "height": 512, "frames": 39, "steps": 20}]
    settings["sample_prompts_data"] = prompts
    assert MODES[-1] == MODE
    fields = {f["key"]: f for section in settings_schema()["sections"] for f in section["fields"]}
    assert all(fields[key]["modes"] == [MODE] for key in MODEL_FIELDS)
    plan = build_command_plan(settings, preview=True)
    cache, text = plan["cache"]
    assert latent_parser(cache_latents.setup_parser_common()).parse_args(cache[2:]).h3_picture_model
    assert text_parser(cache_text_encoder_outputs.setup_parser_common()).parse_args(text[2:]).h3_picture_model
    cmd = plan["train"][0]
    index = cmd.index("src/musubi_tuner/minimax_h3_image_train_network.py")
    args = minimax_h3_image_setup_parser(setup_parser_common()).parse_args(cmd[index + 1:])
    assert args.h3_picture_model and args.h3_training_assistant_enabled and not args.h3_guidance_distillation_protection
    MiniMaxH3PictureNetworkTrainer().handle_model_specific_args(args)
    preview, _ = build_prompt_preview(settings, prompts)
    preview_args = preview_parser().parse_args(preview[2:])
    assert preview_args.h3_picture_model and preview_args.steps == 6 and not preview_args.network_weights
    output = prepare_sample_prompt_settings(settings, write=True)["sample_prompts"]
    from pathlib import Path
    assert "--f 1" in Path(output).read_text(encoding="utf-8")


def test_h3_image_tiny_head_selection_and_picture_decoder():
    model = MiniMaxH3Model(MiniMaxH3Config(
        hidden_size=16, num_layers=1, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=8, ffn_hidden_size=16, text_dim=8, time_embed_hidden_size=16,
        time_embed_dim=8, rope_inv_freq_len=1, picture_model=True,
    ), dtype=torch.float32)
    with torch.no_grad():
        model.rope.inv_freq.fill_(1)
        model.final_layer.video_out.weight.zero_()
        model.final_layer.video_out.bias.copy_(torch.arange(6.0).repeat_interleave(96))
    assert model.audio_patch_proj is None
    x, text = torch.zeros(1, 24, 1, 2, 2), torch.ones(1, 2, 8)
    for head, sigma in enumerate(H3_PICTURE_SIGMAS[:-1]):
        expected = torch.full_like(x, head)
        torch.testing.assert_close(model.forward_image(x, 1 - sigma, text), expected)
    decoder = ViT3DDecoder(patch_size=2, patch_size_t=1, in_channels=24, num_layers=1, heads=2,
                           dim_head=8, rope_dims=2, rope_dim_ratio=0.5)
    assert decoder(x).shape == (1, 3, 1, 4, 4)


def test_h3_picture_sampler_uses_release_grid():
    times = []
    def predict(latent, timestep, text):
        times.append(float(timestep))
        return torch.ones_like(latent)
    sample_image_latent(SimpleNamespace(forward_image=predict), torch.zeros(1, 1, 5120), width=32, height=32,
                        steps=20, seed=17, device="cpu", dtype=torch.float32, picture_model=True)
    assert times == pytest.approx([1 - sigma for sigma in H3_PICTURE_SIGMAS[:-1]])


def test_h3_native_prompt_writes_endpoints_with_spaces(tmp_path):
    first, last = (tmp_path / "first frame.png").resolve(), (tmp_path / "last frame.png").resolve()
    first.touch()
    last.touch()
    config = {
        "training_mode": "MiniMax H3 (Experimental)", "minimax_h3_training_workflow": "Video + audio",
        "minimax_h3_multimodal_task": "fl2va", "output_dir": str(tmp_path),
        "sample_prompts_data": [{"prompt": "Walk", "first_frame": str(first), "last_frame": str(last), "flow_shift": 12}],
    }
    from pathlib import Path
    file = Path(prepare_sample_prompt_settings(config, write=True)["sample_prompts"])
    [prompt] = json.loads(file.read_text(encoding="utf-8"))
    assert prompt["first_frame"] == str(first) and prompt["last_frame"] == str(last)
    assert prompt["frame_count"] == 5 and "discrete_flow_shift" not in prompt


def test_run_control_accepts_one_request_and_acknowledges_saved_stop(tmp_path):
    control = RunControl(tmp_path / "control")
    control.start(can_sample=False)
    with pytest.raises(ValueError, match="sample prompts"):
        control.queue("sample")
    control.queue("save_stop")
    with pytest.raises(ValueError, match="ready"):
        control.queue("save")
    assert control.action() == "save_stop"
    control.complete("save_stop", 7, can_sample=False)
    assert control.status()["status"] == "stopped"
    assert control.status()["step"] == 7 and control.action() is None


def test_corrected_sigma_uses_high_noise_bounds_and_legacy_mapping_remains(monkeypatch):
    from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

    monkeypatch.setattr("musubi_tuner.training.trainer_base.compute_density_for_timestep_sampling",
                        lambda **kwargs: torch.tensor([0.5]))
    args = SimpleNamespace(timestep_sampling="sigma_corrected", weighting_scheme="none", logit_mean=0,
                           logit_std=1, mode_scale=1.29, min_timestep=800, max_timestep=1000)
    eps, data = torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
    scheduler = FlowMatchDiscreteScheduler()
    trainer = NetworkTrainer()
    noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(args, eps, data, None, scheduler, "cpu", torch.float32)
    assert timesteps.item() == pytest.approx(900)
    torch.testing.assert_close(noisy, torch.full_like(noisy, 0.9))
    args.timestep_sampling = "sigma"
    _, old = trainer.get_noisy_model_input_and_timesteps(args, eps, data, None, scheduler, "cpu", torch.float32)
    assert old.item() == pytest.approx(100)


def test_moss_masks_use_sidecar_and_scale_only_selected_rows(tmp_path):
    image = tmp_path / "subject.png"
    image.touch()
    mask = Image.new("L", (32, 32))
    mask.paste(255, (0, 0, 16, 32))
    mask.save(tmp_path / "subject-masklabel.png")
    item = SimpleNamespace(item_key="subject", original_size=(32, 32), bucket_size=(32, 32), latent_cache_path="latent")
    batches = SimpleNamespace(buckets={(32, 32): [item]})
    dataset = SimpleNamespace(image_jsonl_file=None, datasource=SimpleNamespace(image_paths=[image]), batch_manager=batches)
    prepare_masks([dataset], pixels_per_token=16)
    packed = torch.from_numpy(batches.image_loss_masks["latent"])
    assert packed.tolist() == [[255, 0], [255, 0]]
    args = SimpleNamespace(weighting_scheme="none", moss_mask_background_weight=0.0, moss_pretraining_schedule=True)
    y = torch.tensor([[[[[1.0, 5.0], [1.0, 5.0]]]]])
    loss, _ = MossImageNetworkTrainer().compute_loss(args, DiTOutput(y, torch.zeros_like(y)), torch.tensor([250.0]),
                                                     None, None, None, 0, mask=packed.unsqueeze(0))
    assert loss.item() == 1.0


@pytest.mark.parametrize("mode", MASK_MODES)
def test_still_image_mask_switch_reaches_training_command_and_defaults_off(mode, tmp_path):
    settings = {
        "training_mode": mode, "dataset_config": str(tmp_path / "dataset.toml"),
        "output_dir": str(tmp_path), "output_name": "masked", "network_type": "LoRA",
        "train_low_noise": True, "image_mask_background_weight": "0,25",
    }
    unmasked = build_command_plan(settings, preview=True)["train"][0]
    assert "--image_masked_loss" not in unmasked
    settings["image_masked_loss"] = True
    masked = build_command_plan(settings, preview=True)["train"][0]
    assert "--image_masked_loss" in masked
    assert masked[masked.index("--image_mask_background_weight") + 1] == "0.25"
    if mode == "MiniMax H3 (Experimental)":
        settings["minimax_h3_training_workflow"] = "Video + images"
        native = build_command_plan(settings, preview=True)["train"][0]
        assert "--image_masked_loss" not in native


@pytest.mark.parametrize("frame_axis", [False, True])
def test_shared_mask_loss_weights_images_individually(frame_axis):
    squared_error = torch.tensor([[[[1., 9.], [1., 9.]]], [[[3., 3.], [3., 3.]]]])
    masks = torch.tensor([[[255, 0], [255, 0]], [[255, 255], [255, 255]]], dtype=torch.uint8)
    if frame_axis:
        squared_error = squared_error.unsqueeze(2)
    assert mean_image_loss(squared_error, masks, 0).item() == 2.0
    assert mean_image_loss(squared_error, masks, 1).item() == squared_error.mean().item()


def test_network_trainer_masks_gradient_on_black_background():
    prediction = torch.ones(1, 2, 1, 2, 2, requires_grad=True)
    output = DiTOutput(
        prediction, torch.zeros_like(prediction),
        {"image_loss_mask": torch.tensor([[[255, 0], [255, 0]]], dtype=torch.uint8)},
    )
    args = SimpleNamespace(weighting_scheme="none", image_mask_background_weight=0)
    loss, _ = NetworkTrainer().compute_loss(args, output, torch.tensor([500.0]), None, torch.float32, torch.float32, 0)
    loss.backward()
    assert loss.item() == 1.0
    assert prediction.grad[..., 0].abs().min() > 0
    assert prediction.grad[..., 1].count_nonzero() == 0


def test_mask_crops_to_bucket_with_the_model_specific_stride(tmp_path):
    source = tmp_path / "wide.png"
    image = Image.new("L", (64, 32), 255)
    image.paste(0, (24, 0, 48, 32))
    image.save(tmp_path / "wide-masklabel.png")
    item = SimpleNamespace(item_key="wide", original_size=(64, 32), bucket_size=(32, 32), latent_cache_path="latent")
    batches = SimpleNamespace(buckets={(32, 32): [item]})
    dataset = SimpleNamespace(image_jsonl_file=None, datasource=SimpleNamespace(image_paths=[source]), batch_manager=batches)
    prepare_masks([dataset], pixels_per_token=8)
    assert batches.image_loss_masks["latent"].tolist() == [[255, 0, 0, 0]] * 4


def test_ema_writes_new_file_for_ordered_same_run_checkpoints(tmp_path, monkeypatch):
    first, second = tmp_path / "first.safetensors", tmp_path / "second.safetensors"
    for path, value in [(first, 1.0), (second, 3.0)]:
        save_file({"lora.lora_up.weight": torch.tensor([[value]])}, str(path))
    command, destination = plan_ema(f"{first}\n{second}", "", "0,5")
    monkeypatch.setattr(sys, "argv", ["lora-ema", *command[3:]])
    from backends.lora_ema import main
    main()
    assert load_file(destination)["lora.lora_up.weight"].item() == 2.0
    assert load_file(first)["lora.lora_up.weight"].item() == 1.0
