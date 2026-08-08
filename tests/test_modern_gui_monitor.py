from modern_gui.monitor import is_training_progress_line, parse_training_line
from modern_gui.jobs import JobSupervisor


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


def test_only_main_training_bar_is_replaceable_progress():
    assert is_training_progress_line("steps:  25%|##5| 373/1510 [16:50<51:20, 2.71s/it]")
    assert not is_training_progress_line("Loading weights: 25%|##5| 3/12 [00:02<00:06]")
    assert not is_training_progress_line("epoch 1/2")


def test_job_monitor_replaces_transient_progress_and_same_step_loss():
    supervisor = JobSupervisor()
    supervisor._active = {"metrics": {"loss_history": []}}
    supervisor._append_log("system", "training started")
    supervisor._append_log("output", "steps:  1%|#| 1/100 [00:01<01:39, 1.0s/it, avr_loss=0.3]")
    supervisor._append_log("output", "steps:  2%|#| 2/100 [00:02<01:38, 1.0s/it, avr_loss=0.2]")
    supervisor._append_log("output", "steps:  2%|#| 2/100 [00:02<01:38, 1.0s/it, avr_loss=0.19]")

    snapshot = supervisor.snapshot()

    assert [entry["message"] for entry in snapshot["log"]] == [
        "training started",
        "steps:  2%|#| 2/100 [00:02<01:38, 1.0s/it, avr_loss=0.19]",
    ]
    assert supervisor._active["metrics"]["loss_history"] == [[1, 0.3], [2, 0.19]]


def test_job_monitor_discards_empty_carriage_return_output():
    supervisor = JobSupervisor()
    supervisor._active = {"metrics": {"loss_history": []}}

    supervisor._append_log("output", "\r")
    supervisor._append_log("output", "")
    supervisor._append_log("output", "useful output")

    assert [entry["message"] for entry in supervisor.snapshot()["log"]] == ["useful output"]


def test_stop_after_next_epoch_waits_for_the_following_epoch_header():
    class Process:
        def __init__(self):
            self.signals = []

        def poll(self):
            return None

        def send_signal(self, signal):
            self.signals.append(signal)

    supervisor = JobSupervisor()
    process = Process()
    supervisor._active = {
        "status": "running",
        "kind": "training",
        "metrics": {"epoch": 1, "total_epochs": 4},
        "settings": {"max_train_epochs": "4"},
    }
    supervisor._process = process

    armed = supervisor.stop_after_next_epoch(True)
    assert armed["stop_after_epoch"] == 2

    supervisor._append_log("output", "epoch 2/4")
    assert supervisor._active["status"] == "running"
    assert not supervisor._active["stop_after_epoch_triggered"]
    assert process.signals == []

    supervisor._append_log("output", "generating sample images at step / sample: 20")
    supervisor._append_log("output", "epoch 3/4")
    assert supervisor._active["status"] == "stopping"
    assert supervisor._active["stop_after_epoch_triggered"]
    assert supervisor._stop_requested
    assert process.signals
