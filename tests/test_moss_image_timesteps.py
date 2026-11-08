"""Noise distribution, conditioning and legacy recipe compatibility."""
from types import SimpleNamespace

import pytest
import torch

from backends.moss_image import build_commands, training_options
from backends.moss_settings import CHOICES, DEFAULTS, MODE, timestep_schedule
from modern_gui import settings as gui_settings
from musubi_tuner.moss_image.timesteps import sigmoid_high_timesteps
from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, moss_setup_parser, setup_parser_common


HIGH = "Sigmoid + High (AI Toolkit)"


def test_sigmoid_high_matches_reference_pool_and_cubic_indices_and_rng_resume():
    device = torch.device("cpu")
    u = torch.tensor([0., .1, .5, .9, 1.])  # Include both stratification boundaries.
    torch.manual_seed(42)
    pool = ((1 - torch.sigmoid(torch.randn(1000))) * 1000).sort(descending=True).values
    expected = pool[(u ** 3 * 1000).long().clamp(0, 999)]
    torch.manual_seed(42)
    actual = sigmoid_high_timesteps(len(u), device, u)
    torch.testing.assert_close(actual, expected)
    assert actual[0] == pool.max() and actual[-1] == pool.min()
    assert all(actual[:-1] >= actual[1:])

    state = torch.get_rng_state()
    following = sigmoid_high_timesteps(8, device)
    torch.set_rng_state(state)
    torch.testing.assert_close(sigmoid_high_timesteps(8, device), following)
    assert not torch.equal(following, sigmoid_high_timesteps(8, device))


def test_high_bias_and_actual_noise_match_model_conditioning():
    torch.manual_seed(8)
    count = 4096
    clean, noise = torch.zeros(count, 1, 1, 1, 1), torch.ones(count, 1, 1, 1, 1)
    trainer = MossImageNetworkTrainer()
    args = SimpleNamespace(moss_sigmoid_high=True, moss_pretraining_schedule=False, timestep_sampling="uniform")
    noisy, times = trainer.get_noisy_model_input_and_timesteps(args, noise, clean, None, None, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(noisy.flatten(), times / 1000)
    torch.testing.assert_close(trainer._model_timesteps(args, times), times)
    assert 0 < times.min() < times.max() < 1000
    assert .65 < noisy.mean() < .80
    # Same pool with balanced indices has much less high-noise emphasis.
    torch.manual_seed(8)
    pool = ((1 - torch.sigmoid(torch.randn(1000))) * 1000).sort(descending=True).values
    assert times.mean() > pool.mean() + 150
    assert not torch.equal(times, times.round())  # Do not snap this schedule to the pretraining grid.


@pytest.mark.parametrize("selection,legacy,pretraining,high", [
    (None, True, True, False), (None, False, False, False),
    ("", False, False, False), (HIGH, True, False, True),
    ("MOSS pretraining (default)", False, True, False),
    ("General timestep controls", True, False, False),
])
def test_saved_recipe_reaches_exactly_one_schedule(selection, legacy, pretraining, high):
    settings = {**DEFAULTS, "moss_timestep_schedule": selection, "moss_pretraining_schedule": legacy,
                "output_name": "schedule", "output_dir": ".", "timestep_sampling": "uniform"}
    original = dict(settings)
    command = build_commands(settings)[0]
    script = command.index("src/musubi_tuner/moss_image_train_network.py")
    args = moss_setup_parser(setup_parser_common()).parse_args(command[script + 1:])
    assert args.moss_pretraining_schedule is pretraining
    assert args.moss_sigmoid_high is high
    assert args.timestep_sampling == "uniform"
    MossImageNetworkTrainer().handle_model_specific_args(args)
    assert settings == original  # Previewing a command must not rewrite the recipe.


def test_invalid_or_conflicting_schedules_fail_before_loading_models():
    with pytest.raises(ValueError, match="noise schedule"):
        training_options({"moss_timestep_schedule": "misspelled"})
    parser = moss_setup_parser(setup_parser_common())
    args = parser.parse_args(["--moss_pretraining_schedule", "--moss_sigmoid_high"])
    with pytest.raises(ValueError, match="not both"):
        MossImageNetworkTrainer().handle_model_specific_args(args)
    args = parser.parse_args(["--moss_sigmoid_high", "--min_timestep", "100"])
    with pytest.raises(ValueError, match="full noise range"):
        MossImageNetworkTrainer().handle_model_specific_args(args)


@pytest.mark.parametrize("legacy", [True, False])
def test_modern_load_and_schema_preserve_old_checkbox_recipes(tmp_path, monkeypatch, legacy):
    import json
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"training_mode": MODE, "moss_pretraining_schedule": legacy}), encoding="utf-8")
    monkeypatch.setattr(gui_settings, "LAST_SETTINGS", path)
    settings = gui_settings.load_settings()
    assert settings["moss_timestep_schedule"] == ("MOSS pretraining (default)" if legacy else "General timestep controls")
    schema = gui_settings.settings_schema(settings)
    fields = {field["key"]: field for section in schema["sections"] for field in section["fields"]}
    assert "moss_pretraining_schedule" not in fields
    field = fields["moss_timestep_schedule"]
    assert field["type"] == "select" and field["options"] == CHOICES["moss_timestep_schedule"]
    assert field["modes"] == [MODE] and not field["allow_custom"]
    assert "Sigmoid + High" in field["help_detail"] and "recaching" in field["help_detail"]


def test_schedule_switch_keeps_cache_commands_and_other_options_unchanged():
    from backends.moss_image import build_cache_commands
    settings = {**DEFAULTS, "dataset_config": "dataset.toml", "vae_model": "vae.safetensors",
                "recache_latents": True, "recache_text": True}
    before = build_cache_commands(settings, "python")
    settings["moss_timestep_schedule"] = HIGH
    assert build_cache_commands(settings, "python") == before
    assert timestep_schedule(training_options(settings)) == HIGH
