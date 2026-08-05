from pathlib import Path

from PIL import Image

from backends._common import build_sample_args
from modern_gui.sampling import estimate_steps_per_epoch, fractional_epoch_to_steps


def _dataset(tmp_path: Path) -> Path:
    source = tmp_path / "images"
    source.mkdir()
    for index in range(5):
        image_path = source / f"image_{index}.png"
        Image.new("RGB", (64, 64), (index, index, index)).save(image_path)
        image_path.with_suffix(".txt").write_text("portrait", encoding="utf-8")
    config = tmp_path / "dataset.toml"
    config.write_text(
        f'[general]\nresolution = [64, 64]\ncaption_extension = ".txt"\nbatch_size = 2\n\n'
        f'[[datasets]]\nimage_directory = "{source.as_posix()}"\nnum_repeats = 1\n',
        encoding="utf-8",
    )
    return config


def test_estimate_steps_per_epoch_accounts_for_batch_and_accumulation(tmp_path):
    config = _dataset(tmp_path)

    estimate = estimate_steps_per_epoch(str(config), 2)

    assert estimate["effective_samples"] == 5
    assert estimate["batches_per_epoch"] == 3
    assert estimate["steps_per_epoch"] == 2
    assert fractional_epoch_to_steps("0.5", str(config), 2) == 1


def test_fractional_epoch_sampling_becomes_step_sampling(tmp_path):
    config = _dataset(tmp_path)
    command = []

    build_sample_args(
        command,
        {
            "dataset_config": str(config),
            "sample_prompts": "prompts.txt",
            "sample_every_n_epochs": "0.5",
            "gradient_accumulation_steps": "2",
        },
    )

    assert "--sample_every_n_epochs" not in command
    assert command[command.index("--sample_every_n_steps") + 1] == "1"
