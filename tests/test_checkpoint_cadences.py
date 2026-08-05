from backends._common import build_common_train_args


def test_zero_checkpoint_cadences_are_omitted_without_affecting_other_args():
    command = []

    build_common_train_args(
        command,
        {
            "learning_rate": "1e-4",
            "save_every_n_epochs": "0",
            "save_every_n_steps": "543",
        },
    )

    assert "--save_every_n_epochs" not in command
    assert command[command.index("--save_every_n_steps") + 1] == "543"


def test_zero_step_checkpoint_cadence_is_omitted_independently():
    command = []

    build_common_train_args(
        command,
        {
            "save_every_n_epochs": "1",
            "save_every_n_steps": 0,
        },
    )

    assert command[command.index("--save_every_n_epochs") + 1] == "1"
    assert "--save_every_n_steps" not in command
