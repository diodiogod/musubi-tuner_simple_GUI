from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from job_performance import (
    append_job_log,
    compact_speed_history,
    continuation_metadata,
    job_log_path,
    parse_training_speed,
)

from modern_gui.commands import build_command_plan
from modern_gui.training_notes import effective_training_comment
from modern_gui.stages import (
    prepare_standard_stage,
    resolve_standard_state,
    stage_label,
    validate_stage_plan,
)
from modern_gui.monitor import is_training_progress_line, parse_training_line
from modern_gui.face_stages import prepare_face_stage, validate_face_environment
from modern_gui.stages import resolve_stage_lora
from process_lifecycle import ProcessTreeScope


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "modern_gui_jobs.json"
RANK_ENVIRONMENT_KEYS = {
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _subprocess_environment() -> dict[str, str]:
    """Build a clean UTF-8 environment for Musubi and Accelerate children."""

    environment = os.environ.copy()
    for key in RANK_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    # Windows otherwise gives redirected Python stdout the active ANSI code
    # page (often CP1252), which cannot represent Musubi's bilingual logs.
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _read_history() -> list[dict[str, Any]]:
    try:
        payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _write_history(jobs: list[dict[str, Any]]) -> None:
    temporary = HISTORY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(jobs[:200], indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(HISTORY_PATH)


def _lineage_history(base: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Include both modern and classic records when finding a parent run."""

    history = list(base) if base is not None else _read_history()
    try:
        # Imported lazily to keep the recovery module independent from the
        # supervisor's process/thread setup during application startup.
        from modern_gui.recovery import load_desktop_history

        history.extend(load_desktop_history())
    except (ImportError, OSError, ValueError):
        pass
    return history


def _history_with_lineage() -> list[dict[str, Any]]:
    """Backfill lineage in chronological order, including older child runs."""

    jobs = _read_history()
    pool = _lineage_history(jobs)
    for job in sorted(jobs, key=lambda item: str(item.get("started_at") or "")):
        settings = job.get("settings") if isinstance(job.get("settings"), dict) else job.get("settings_snapshot")
        if isinstance(settings, dict):
            # Repair display data for runs created before the effective
            # automatic note was copied into the history snapshot.
            if settings.get("auto_training_settings_summary") and not str(settings.get("training_comment") or "").strip():
                settings["training_comment"] = effective_training_comment(settings)
            if not any(job.get(key) for key in ("continuation_prior_steps", "continuation_prior_epochs", "continuation_parent_id")):
                job.update(continuation_metadata(settings, pool))
    return jobs


def _history_pool_with_lineage() -> list[dict[str, Any]]:
    jobs = _history_with_lineage()
    try:
        from modern_gui.recovery import load_desktop_history

        return jobs + load_desktop_history()
    except (ImportError, OSError, ValueError):
        return jobs


class JobSupervisor:
    def __init__(self):
        self._lock = threading.RLock()
        self._active: dict[str, Any] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._process_scope: ProcessTreeScope | None = None
        self._log: deque[dict[str, Any]] = deque(maxlen=5000)
        self._next_log_id = 1
        self._stop_requested = False

    def history(self) -> list[dict[str, Any]]:
        # Older modern records predate explicit lineage fields. Derive them on
        # read so their history cards benefit too; the original JSON is left
        # untouched and new runs persist the metadata at start time.
        return _history_with_lineage()

    def clear_history(self) -> None:
        with self._lock:
            if self._active and self._active.get("status") in {"starting", "running", "stopping"}:
                raise RuntimeError("Job history cannot be cleared while a process is active.")
            _write_history([])

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        with self._lock:
            active = dict(self._active) if self._active else None
            log = [entry for entry in self._log if entry["id"] > after]
            return {"active": active, "log": log, "last_log_id": self._next_log_id - 1}

    def start(self, settings: dict[str, Any], run_cache: bool = True) -> dict[str, Any]:
        with self._lock:
            if self._active and self._active.get("status") in {"starting", "running", "stopping"}:
                raise RuntimeError("A web GUI job is already active.")
            staged = bool(settings.get("use_staged_training"))
            stages = validate_stage_plan(settings) if staged else []
            if any(stage.get("type", "standard") == "face_refinement" for stage in stages):
                validate_face_environment(settings.get("face_refinement_config") or {})
            if staged:
                commands = []
            else:
                plan = build_command_plan(settings)
                commands = (plan["cache"] if run_cache else []) + plan["train"]
                if not commands:
                    raise ValueError("The current settings did not produce any commands.")
            self._log.clear()
            self._next_log_id = 1
            self._stop_requested = False
            job_id = str(uuid.uuid4())
            console_log = job_log_path(job_id)
            console_log.unlink(missing_ok=True)
            lineage = continuation_metadata(settings, _history_pool_with_lineage())
            # ``build_command_plan`` works on a private copy and injects the
            # generated settings summary there.  Keep the same effective note
            # in the history snapshot, otherwise Files & notes shows an empty
            # comment even though the trainer received one.
            history_settings = deepcopy(settings)
            history_settings["training_comment"] = effective_training_comment(history_settings)
            self._active = {
                "id": job_id,
                "name": settings.get("output_name") or "unnamed",
                "mode": settings.get("training_mode", "Wan 2.2"),
                "status": "starting",
                "started_at": _utc_now(),
                "finished_at": None,
                "phase": "preparing",
                "command_index": 0,
                "command_count": len(commands) if not staged else len(stages),
                "return_code": None,
                "kind": "staged_training" if staged else "training",
                "metrics": {
                    "step": 0,
                    "total_steps": 0,
                    "epoch": 0,
                    "total_epochs": 0,
                    "loss": None,
                    "depth_loss": None,
                    "dop_loss": None,
                    "dop_weighted": None,
                    "loss_history": [],
                    "speed_history": [],
                },
                "stop_after_epoch": None,
                "stop_after_epoch_triggered": False,
                "settings": history_settings,
                "console_log_path": str(console_log),
            }
            self._active.update(lineage)
            target = self._run_staged if staged else self._run
            arguments = (settings, stages) if staged else (commands,)
            thread = threading.Thread(target=target, args=arguments, daemon=True, name="musubi-web-job")
            thread.start()
            return dict(self._active)

    def start_commands(
        self,
        commands: list[list[str]],
        *,
        name: str,
        mode: str,
        kind: str,
        settings: dict[str, Any],
        completion_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run an existing GUI utility command plan through the shared monitor/history path."""
        if not commands:
            raise ValueError("The utility did not produce any commands.")
        with self._lock:
            if self._active and self._active.get("status") in {"starting", "running", "stopping"}:
                raise RuntimeError("A web GUI job is already active.")
            self._log.clear()
            self._next_log_id = 1
            self._stop_requested = False
            job_id = str(uuid.uuid4())
            console_log = job_log_path(job_id)
            console_log.unlink(missing_ok=True)
            lineage = continuation_metadata(settings, _history_pool_with_lineage()) if kind in {"training", "staged_training"} else {}
            history_settings = deepcopy(settings)
            history_settings["training_comment"] = effective_training_comment(history_settings)
            self._active = {
                "id": job_id,
                "name": name,
                "mode": mode,
                "status": "starting",
                "started_at": _utc_now(),
                "finished_at": None,
                "phase": "preparing",
                "command_index": 0,
                "command_count": len(commands),
                "return_code": None,
                "kind": kind,
                "metrics": {
                    "step": 0, "total_steps": 0, "epoch": 0, "total_epochs": 0,
                    "loss": None, "depth_loss": None, "dop_loss": None,
                    "dop_weighted": None, "loss_history": [], "speed_history": [],
                },
                "stop_after_epoch": None,
                "stop_after_epoch_triggered": False,
                "settings": history_settings,
                "console_log_path": str(console_log),
                "_completion_context": dict(completion_context or {}),
            }
            self._active.update(lineage)
            thread = threading.Thread(target=self._run, args=(commands,), daemon=True, name="musubi-web-utility")
            thread.start()
            return dict(self._active)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._active or self._active.get("status") not in {"starting", "running"}:
                raise RuntimeError("No web GUI job is currently running.")
            self._stop_requested = True
            self._active["status"] = "stopping"
            process = self._process
            if self._process_scope:
                self._process_scope.close()
            if process and process.poll() is None:
                try:
                    if os.name == "nt":
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        process.terminate()
                except (OSError, ValueError):
                    process.terminate()
            return dict(self._active)

    def stop_after_next_epoch(self, enabled: bool = True) -> dict[str, Any]:
        """Arm or disarm a graceful stop after the next completed epoch.

        The trainer emits the next epoch header only after its previous epoch's
        checkpoint and scheduled samples have finished. Watching that boundary
        lets the GUI stop without interrupting an in-progress sample.
        """

        with self._lock:
            if not self._active or self._active.get("status") not in {"starting", "running"}:
                raise RuntimeError("No web GUI training job is currently running.")
            if self._active.get("kind") not in {"training", "staged_training"}:
                raise RuntimeError("Stop-after-next-epoch is available only for training jobs.")
            if enabled:
                metrics = self._active.get("metrics") or {}
                current_epoch = int(metrics.get("epoch") or 0)
                total_epochs = int(metrics.get("total_epochs") or self._active.get("settings", {}).get("max_train_epochs") or 0)
                target = max(1, current_epoch + 1)
                if total_epochs and target > total_epochs:
                    raise RuntimeError("The run is already in its final epoch and will finish naturally.")
                self._active["stop_after_epoch"] = target
                self._active["stop_after_epoch_triggered"] = False
            else:
                self._active["stop_after_epoch"] = None
                self._active["stop_after_epoch_triggered"] = False
            return dict(self._active)

    def _append_log(self, stream: str, message: str) -> None:
        # tqdm and redirected Windows streams can emit bare carriage-return
        # progress updates. After universal-newline handling those become
        # empty output lines; keeping them makes the live terminal grow with
        # invisible rows even though the compact progress header is correct.
        message = str(message).replace("\r", "")
        if stream == "output" and not message.strip():
            return
        with self._lock:
            observed_at = _utc_now()
            transient = stream == "output" and is_training_progress_line(message)
            if transient and self._log and self._log[-1].get("transient"):
                self._log.pop()
            self._log.append({"id": self._next_log_id, "stream": stream, "message": message, "time": observed_at, "transient": transient})
            self._next_log_id += 1
            if self._active:
                try:
                    append_job_log(self._active.get("console_log_path", ""), observed_at, stream, message)
                except OSError:
                    pass
            if stream == "output" and self._active:
                update = parse_training_line(message)
                metrics = self._active.get("metrics", {})
                metrics.update(update)
                stop_after_epoch = self._active.get("stop_after_epoch")
                if stop_after_epoch and "epoch" in update and int(update["epoch"]) > int(stop_after_epoch):
                    # An epoch header is printed after the prior epoch's
                    # checkpoint and scheduled sample generation. Stop here,
                    # before the first optimization step of the following
                    # epoch, so the saved state is resumable.
                    self._active["stop_after_epoch_triggered"] = True
                    self._stop_requested = True
                    self._active["status"] = "stopping"
                    process = self._process
                    if process and process.poll() is None:
                        try:
                            if os.name == "nt":
                                process.send_signal(signal.CTRL_BREAK_EVENT)
                            else:
                                process.terminate()
                        except (OSError, ValueError):
                            process.terminate()
                speed = parse_training_speed(message)
                if speed:
                    history = metrics.setdefault("speed_history", [])
                    history.append([speed["step"], speed["seconds_per_iteration"], observed_at])
                    metrics["speed_history"] = compact_speed_history(history)
                    metrics["seconds_per_iteration"] = speed["seconds_per_iteration"]
                if "loss" in update:
                    history = metrics.setdefault("loss_history", [])
                    point = [int(update.get("step", metrics.get("step", 0))), update["loss"]]
                    if history and history[-1][0] == point[0]:
                        history[-1] = point
                    else:
                        history.append(point)
                    if len(history) > 1200:
                        metrics["loss_history"] = history[::2][-1000:]

    def _read_process_output(self, process: subprocess.Popen[str], job_id: str) -> None:
        """Drain child output without making job completion depend on pipe EOF."""

        if process.stdout is None:
            return
        try:
            for line in process.stdout:
                with self._lock:
                    if not self._active or self._active.get("id") != job_id:
                        continue
                self._append_log("output", line.rstrip())
        except (OSError, ValueError):
            # A stopped process or late-closing helper may invalidate the pipe.
            pass

    def _run(self, commands: list[list[str]]) -> None:
        return_code = self._execute_commands(commands)
        self._finish(return_code)

    def _execute_commands(self, commands: list[list[str]], phase_prefix: str = "") -> int:
        return_code = 0
        try:
            for index, command in enumerate(commands):
                with self._lock:
                    if self._stop_requested:
                        break
                    self._active["status"] = "running"
                    if self._active.get("kind") != "staged_training":
                        self._active["command_index"] = index + 1
                    phase = self._phase(command)
                    self._active["phase"] = f"{phase_prefix} · {phase}" if phase_prefix else phase
                    job_id = str(self._active["id"])
                self._append_log("system", f"$ {subprocess.list2cmdline([str(value) for value in command])}")
                environment = _subprocess_environment()
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                self._process = subprocess.Popen(
                    [str(value) for value in command],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=environment,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt",
                )
                self._process_scope = ProcessTreeScope(self._process)
                output_reader = threading.Thread(
                    target=self._read_process_output,
                    args=(self._process, job_id),
                    daemon=True,
                    name=f"musubi-web-output-{job_id[:8]}",
                )
                output_reader.start()
                return_code = self._process.wait()
                self._process_scope.close()
                # W&B and similar helpers may inherit stdout and keep the pipe
                # open after Accelerate exits. Do not leave the job stuck in
                # "stopping" while waiting for those unrelated handles.
                output_reader.join(timeout=0.5)
                self._process = None
                self._process_scope = None
                if return_code != 0:
                    break
        except Exception as exc:
            return_code = -1
            self._append_log("error", f"{type(exc).__name__}: {exc}")
            if self._process_scope:
                self._process_scope.close()
            self._process = None
            self._process_scope = None
        return return_code

    def _run_staged(self, base_settings: dict[str, Any], stages: list[dict[str, Any]]) -> None:
        return_code = 0
        previous_settings: dict[str, Any] | None = None
        previous_type: str | None = None
        previous_face_output: Path | None = None
        try:
            for index, stage in enumerate(stages):
                if self._stop_requested:
                    break
                current_type = stage.get("type", "standard")
                if current_type == "standard":
                    fresh_optimizer = (
                        previous_settings is not None and previous_type == "standard"
                        and str(stage.get("handoff_mode", "state")) == "weights"
                    )
                    resume_path = (
                        str(resolve_standard_state(previous_settings))
                        if previous_settings and previous_type == "standard" and not fresh_optimizer
                        else ""
                    )
                    network_weights = (
                        str(previous_face_output)
                        if previous_type == "face_refinement"
                        else str(resolve_stage_lora(previous_settings))
                        if fresh_optimizer and previous_settings
                        else str(base_settings.get("network_weights") or "")
                        if index == 0
                        else ""
                    )
                    stage_settings = prepare_standard_stage(
                        base_settings,
                        stage,
                        index,
                        resume_path=resume_path,
                        network_weights=network_weights,
                    )
                    plan = build_command_plan(stage_settings)
                    commands = plan["cache"] + plan["train"]
                    expected_face_output = None
                else:
                    if previous_type == "standard" and previous_settings:
                        input_lora = resolve_stage_lora(previous_settings)
                    elif previous_type == "face_refinement" and previous_face_output:
                        input_lora = previous_face_output
                    else:
                        configured = str((base_settings.get("face_refinement_config") or {}).get("input_lora", ""))
                        input_lora = Path(configured).expanduser()
                    stage_settings, face_command, expected_face_output = prepare_face_stage(
                        base_settings, stage, index, input_lora
                    )
                    commands = [face_command]
                with self._lock:
                    self._active["command_index"] = index + 1
                    self._active["phase"] = f"Stage {index + 1}/{len(stages)} · {stage_label(stage, index)}"
                    self._active["stage_settings"] = stage_settings
                self._append_log(
                    "system",
                    f"=== Stage {index + 1}/{len(stages)}: {stage_label(stage, index)} "
                    f"· {'steps ' + stage_settings['max_train_steps'] if stage_settings['max_train_steps'] else 'epochs ' + stage_settings['max_train_epochs']} ===",
                )
                return_code = self._execute_commands(
                    commands, f"Stage {index + 1}/{len(stages)} · {stage_label(stage, index)}"
                )
                if return_code != 0 or self._stop_requested:
                    break
                if expected_face_output is not None and not expected_face_output.is_file():
                    raise FileNotFoundError(
                        f"Face Refinement finished without creating its expected LoRA: {expected_face_output}"
                    )
                self._record_completed_stage(
                    index,
                    stage,
                    stage_settings,
                    expected_face_output,
                )
                previous_settings = stage_settings
                previous_type = current_type
                previous_face_output = expected_face_output
        except Exception as exc:
            return_code = -1
            self._append_log("error", f"{type(exc).__name__}: {exc}")
        self._finish(return_code)

    def _record_completed_stage(
        self,
        index: int,
        stage: dict[str, Any],
        stage_settings: dict[str, Any],
        expected_face_output: Path | None,
    ) -> None:
        """Keep compact handoff lineage plus the complete final-stage recipe."""

        stage_type = str(stage.get("type") or "standard")
        artifacts = {"state": "", "lora": ""}
        if stage_type == "standard":
            try:
                artifacts["state"] = str(resolve_standard_state(stage_settings))
            except FileNotFoundError:
                pass
            try:
                artifacts["lora"] = str(resolve_stage_lora(stage_settings))
            except FileNotFoundError:
                pass
        elif expected_face_output is not None:
            artifacts["lora"] = str(expected_face_output)

        lineage_entry = {
            "index": index,
            "label": stage_label(stage, index),
            "type": stage_type,
            "output_name": str(stage_settings.get("output_name") or ""),
            "dataset_config": str(stage_settings.get("dataset_config") or ""),
            "max_train_epochs": str(stage_settings.get("max_train_epochs") or ""),
            "max_train_steps": str(stage_settings.get("max_train_steps") or ""),
            "artifacts": artifacts,
        }
        with self._lock:
            if not self._active:
                return
            self._active.setdefault("stage_lineage", []).append(lineage_entry)
            self._active["final_stage_settings"] = deepcopy(stage_settings)
            self._active["final_stage_artifacts"] = dict(artifacts)

    def _finish(self, return_code: int) -> None:
        captured = 0
        sample_outputs: list[str] = []
        with self._lock:
            context = dict((self._active or {}).get("_completion_context") or {})
            final_settings = deepcopy((self._active or {}).get("settings") or {})
            staged = (self._active or {}).get("kind") == "staged_training"
        if return_code == 0 and context.get("kind") == "sample_test":
            sample_outputs = self._find_new_sample_outputs(context)
            captured = self._capture_sample_test_thumbnails(context)
        if return_code == 0 and not staged:
            for message in self._rename_final_training_artifacts(final_settings):
                self._append_log("system", message)
        with self._lock:
            stopped = self._stop_requested
            if self._active:
                self._active["status"] = "stopped" if stopped else ("completed" if return_code == 0 else "failed")
                self._active["finished_at"] = _utc_now()
                self._active["return_code"] = return_code
                self._active["phase"] = "finished"
                if sample_outputs:
                    self._active["sample_outputs"] = sample_outputs
                if captured:
                    self._active["captured_thumbnails"] = captured
                record = {key: value for key, value in self._active.items() if key not in {"settings", "stage_settings", "_completion_context"}}
                # Staged execution mutates only separate stage snapshots. Keep the
                # original recipe intact and store the final effective stage beside it.
                record["settings"] = deepcopy(self._active["settings"])
                history = _read_history()
                history.insert(0, record)
                _write_history(history)
            self._process = None

    @staticmethod
    def _rename_final_training_artifacts(settings: dict[str, Any]) -> list[str]:
        """Mirror the Classic GUI's optional final epoch naming behavior."""

        if not settings.get("rename_final_artifacts_to_epoch", True):
            return []
        output_dir = str(settings.get("output_dir") or "").strip()
        output_name = str(settings.get("output_name") or "").strip()
        epoch_text = str(settings.get("max_train_epochs") or "").strip()
        if not output_dir or not output_name or not epoch_text.isdigit():
            return []
        run_dir = Path(output_dir) / output_name
        if not run_dir.is_dir():
            return []
        suffix = f"{int(epoch_text):06d}"
        pairs = [
            (run_dir / f"{output_name}.safetensors", run_dir / f"{output_name}-{suffix}.safetensors"),
            (run_dir / f"{output_name}-state", run_dir / f"{output_name}-{suffix}-state"),
        ]
        messages = []
        for source, target in pairs:
            if not source.exists():
                continue
            if target.exists():
                messages.append(f"Final artifact rename skipped; target already exists: {target.name}")
                continue
            try:
                source.rename(target)
                messages.append(f"Final artifact renamed: {source.name} -> {target.name}")
            except OSError as exc:
                messages.append(f"Final artifact rename failed for {source.name}: {exc}")
        return messages

    @staticmethod
    def _find_new_sample_outputs(context: dict[str, Any]) -> list[str]:
        save_path = Path(str(context.get("save_path") or ""))
        before = set(context.get("existing_outputs") or [])
        supported = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov", ".m4v"}
        outputs = [
            path for path in save_path.glob("*")
            if path.is_file() and path.suffix.lower() in supported and str(path.resolve()) not in before
        ]
        outputs.sort(key=lambda path: (path.stat().st_mtime, path.name))
        return [str(path.resolve()) for path in outputs]

    @staticmethod
    def _capture_sample_test_thumbnails(context: dict[str, Any]) -> int:
        from prompt_library import PromptLibraryStore

        save_path = Path(str(context.get("save_path") or ""))
        before = set(context.get("existing_outputs") or [])
        images = [
            path for path in save_path.glob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and str(path.resolve()) not in before
        ]
        images.sort(key=lambda path: (path.stat().st_mtime, path.name))
        store = PromptLibraryStore()
        captured = 0
        for prompt, image_path in zip(context.get("prompts") or [], images):
            entry, _created = store.capture_thumbnail(
                prompt,
                image_path,
                str(context.get("mode") or "Krea 2"),
                metadata={
                    "test_type": "standalone_zero_step",
                    "output_name": context.get("output_name", ""),
                    "network_weights": context.get("network_weights", ""),
                    "seed": prompt.get("seed", ""),
                    "width": prompt.get("width", ""),
                    "height": prompt.get("height", ""),
                },
            )
            captured += int(entry is not None)
        return captured

    @staticmethod
    def _phase(command: list[str]) -> str:
        joined = " ".join(str(value) for value in command).lower()
        if "cache_latents" in joined:
            return "Caching latents"
        if "cache_text_encoder" in joined:
            return "Caching text"
        if "face_refinement" in joined:
            return "Face refinement"
        return "Training"


SUPERVISOR = JobSupervisor()
