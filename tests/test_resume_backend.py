from backends._common import build_common_train_args
from musubi_tuner_gui import MusubiTunerGUI


def test_ordinary_continuation_does_not_enable_exact_position():
    settings = {"resume_path": "run-000004-state"}
    command = []

    build_common_train_args(command, settings)

    assert "--resume" in command
    assert "--resume_exact_position" not in command
    assert MusubiTunerGUI._should_use_exact_resume(settings) is False


def test_verified_recovery_enables_exact_position_only_for_matching_state():
    settings = {"resume_path": "run-000004-state", "resume_exact_position": True}
    command = []

    build_common_train_args(command, settings)

    assert "--resume_exact_position" in command
    assert MusubiTunerGUI._should_use_exact_resume(
        settings,
        pending_recovery={"state_path": "run-000004-state"},
    ) is True
    assert MusubiTunerGUI._should_use_exact_resume(
        settings,
        pending_recovery={"state_path": "different-000004-state"},
    ) is False
