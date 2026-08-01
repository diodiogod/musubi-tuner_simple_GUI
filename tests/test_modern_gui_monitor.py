from modern_gui.monitor import parse_training_line


def test_parser_ignores_unrelated_model_loading_bar():
    line = "Loading krea2_turbo.safetensors: 100%|##########| 686/686 [00:23<00:00, 29.13key/s]"
    assert parse_training_line(line) == {}


def test_parser_extracts_training_depth_and_dop_metrics():
    line = (
        "steps:  14%|#4| 47/329 [01:47<10:43, 2.28s/it, "
        "avr_loss=0.119, loss/depth_anchor=0.238, loss/dop=0.044, loss/dop_weighted=0.011]"
    )

    parsed = parse_training_line(line)

    assert parsed == {
        "step": 47,
        "total_steps": 329,
        "loss": 0.119,
        "depth_loss": 0.238,
        "dop_loss": 0.044,
        "dop_weighted": 0.011,
    }


def test_parser_extracts_epochs_and_face_steps():
    assert parse_training_line("epoch 3/10") == {"epoch": 3, "total_epochs": 10}
    assert parse_training_line("step=12/80 loss=0.22") == {"step": 12, "total_steps": 80}
