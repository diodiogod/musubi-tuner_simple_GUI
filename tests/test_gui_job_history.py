from musubi_tuner_gui import MusubiTunerGUI


def test_job_display_name_prefers_recorded_output_name():
    job = {"output_name": "character_v4", "title": "Training run"}
    assert MusubiTunerGUI._job_display_name(job) == "character_v4"


def test_job_display_name_supports_older_settings_snapshot():
    job = {
        "output_name": "",
        "settings_snapshot": {"output_name": "older_run"},
        "title": "Krea Turbo face evaluation",
    }
    assert MusubiTunerGUI._job_display_name(job) == "older_run"


def test_job_display_name_has_readable_fallback():
    job = {"output_dir": "C:/models/example_run", "title": "Training run"}
    assert MusubiTunerGUI._job_display_name(job) == "example_run"
