"""Local command mailbox for one GUI training process."""

import json
from pathlib import Path
from uuid import uuid4


ENV_KEY = "MUSUBI_RUN_CONTROL_DIR"
COMMANDS = frozenset({"sample", "save", "save_state", "save_stop"})


def _read(path):
    try:
        if path.stat().st_size > 4096:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path, data):
    temp = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(data), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def for_training_command(command, log_path):
    if not any(
        str(part).replace("\\", "/").startswith("src/musubi_tuner/") and str(part).endswith("_train_network.py")
        for part in command
    ):
        return None
    log_path = Path(log_path)
    return RunControl(log_path.parent / log_path.stem / uuid4().hex)


class RunControl:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.status_path = self.directory / "status.json"
        self.request_path = self.directory / "request.json"

    def status(self):
        return _read(self.status_path)

    def gui_status(self):
        status = self.status()
        if self.request_path.exists():
            status["status"] = "pending"
        return status

    def start(self, can_sample):
        self.directory.mkdir(parents=True, exist_ok=True)
        _write(self.status_path, {"status": "ready", "can_sample": bool(can_sample)})

    def queue(self, command):
        status = self.status()
        if not isinstance(command, str) or command not in COMMANDS:
            raise ValueError("Unknown run action.")
        if status.get("status") != "ready" or self.request_path.exists():
            raise ValueError("Training is not ready for another action.")
        if command == "sample" and not status.get("can_sample"):
            raise ValueError("Configure training sample prompts before starting the run.")
        _write(self.request_path, {"action": command})

    def action(self):
        request = _read(self.request_path)
        action = request.get("action")
        return action if isinstance(action, str) and action in COMMANDS else None

    def complete(self, action, step, *, can_sample, success=True):
        stopped = action == "save_stop" and success
        _write(self.status_path, {"status": "stopped" if stopped else "ready", "can_sample": bool(can_sample),
                                  "last_action": action, "step": step, "success": success})
        self.request_path.unlink(missing_ok=True)
