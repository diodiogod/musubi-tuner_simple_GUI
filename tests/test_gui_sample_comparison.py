from sample_gallery import parse_training_sample_path


def test_training_sample_parser_groups_epochs_for_same_prompt_and_seed(tmp_path):
    first = parse_training_sample_path(tmp_path / "portrait_e000001_02_20260714120000_123.png")
    later = parse_training_sample_path(tmp_path / "portrait_e000006_02_20260714130000_123.png")

    assert first["group_key"] == later["group_key"]
    assert first["sequence_label"] == "Epoch 1"
    assert later["sequence_label"] == "Epoch 6"
    assert later["prompt_index"] == 2


def test_training_sample_parser_separates_prompts_and_seeds(tmp_path):
    base = parse_training_sample_path(tmp_path / "run_e000004_00_20260714120000_10.webp")
    other_prompt = parse_training_sample_path(tmp_path / "run_e000004_01_20260714120000_10.webp")
    other_seed = parse_training_sample_path(tmp_path / "run_e000004_00_20260714120000_11.webp")

    assert base["group_key"] != other_prompt["group_key"]
    assert base["group_key"] != other_seed["group_key"]


def test_training_sample_parser_supports_step_samples_and_ignores_other_media(tmp_path):
    sample = parse_training_sample_path(tmp_path / "run_000250_03_20260714120000.mp4")

    assert sample["sequence_kind"] == "step"
    assert sample["sequence"] == 250
    assert sample["sequence_label"] == "Step 250"
    assert parse_training_sample_path(tmp_path / "reference.png") is None
