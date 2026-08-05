from modern_gui.sample_prompts import prepare_sample_prompt_settings


def test_minimax_video_card_writes_one_frame_for_scheduled_training(tmp_path):
    settings = {
        "training_mode": "MiniMax H3 (Experimental)",
        "dataset_config": str(tmp_path / "dataset.toml"),
        "output_dir": str(tmp_path),
        "output_name": "h3",
        "sample_prompts_data": [
            {"enabled": True, "prompt": "portrait", "width": 768, "height": 768, "frames": 5}
        ],
    }

    prepared = prepare_sample_prompt_settings(settings, write=True)

    serialized = (tmp_path / "h3_sample_prompts.txt").read_text(encoding="utf-8")
    assert prepared["sample_prompts"].endswith("h3_sample_prompts.txt")
    assert "--f 1" in serialized
    assert settings["sample_prompts_data"][0]["frames"] == 5
