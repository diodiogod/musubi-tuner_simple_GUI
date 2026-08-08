from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from continuation_names import continuation_name
from modern_gui.stages import (
    candidate_lora_paths,
    candidate_state_paths,
    effective_run_name,
    enabled_stages,
    prepare_standard_stage,
    stage_label,
)


ROOT = Path(__file__).resolve().parents[1]
DESKTOP_HISTORY_PATH = ROOT / "job_history_local.json"


def load_desktop_history() -> list[dict[str, Any]]:
    try:
        payload = json.loads(DESKTOP_HISTORY_PATH.read_text(encoding="utf-8"))
        jobs = payload if isinstance(payload, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        jobs = []
    result = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            continue
        item = dict(job)
        item["_source"] = "desktop"
        item["_history_index"] = index
        result.append(item)
    return result


def import_output_jobs(settings: dict[str, Any]) -> dict[str, int]:
    """Import run folders not already represented in desktop history."""
    output_root = Path(str(settings.get("output_dir") or "")).expanduser()
    if not output_root.is_dir():
        raise ValueError(f"Configured output directory does not exist: {output_root}")
    existing = load_desktop_history()
    known = {
        str((job.get("settings_snapshot") or {}).get("output_name") or job.get("output_name") or job.get("name") or "")
        for job in existing
    }
    discovered = []
    for run_dir in sorted((path for path in output_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True):
        if run_dir.name in known:
            continue
        artifacts = [path for path in run_dir.glob("*.safetensors") if path.is_file()]
        states = [path for path in run_dir.glob("*-state") if path.is_dir()]
        if not artifacts and not states and not (run_dir / "sample").is_dir():
            continue
        snapshot = copy.deepcopy(settings)
        snapshot["output_name"] = run_dir.name
        snapshot["output_dir"] = str(output_root)
        discovered.append(
            {
                "job_id": f"discovered-{run_dir.name.casefold()}",
                "kind": "training",
                "title": run_dir.name,
                "output_name": run_dir.name,
                "status": "discovered",
                "started_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
                "finished_at": "",
                "note": "Imported from an existing output folder.",
                "settings_snapshot": snapshot,
                "commands": [],
            }
        )
    if discovered:
        current = json.loads(DESKTOP_HISTORY_PATH.read_text(encoding="utf-8")) if DESKTOP_HISTORY_PATH.is_file() else []
        temporary = DESKTOP_HISTORY_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps((discovered + current)[:200], indent=2), encoding="utf-8")
        temporary.replace(DESKTOP_HISTORY_PATH)
    return {"added": len(discovered), "scanned": len(list(output_root.iterdir()))}


def clear_desktop_history() -> None:
    temporary = DESKTOP_HISTORY_PATH.with_suffix(".json.tmp")
    temporary.write_text("[]\n", encoding="utf-8")
    temporary.replace(DESKTOP_HISTORY_PATH)


def state_epoch(path: Path | str) -> int:
    match = re.search(r"-(\d+)-state$", Path(path).name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def state_step(path: Path | str) -> int:
    match = re.search(r"-step(\d+)-state$", Path(path).name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def validate_accelerate_state(path: Path | str, exact_position: bool = True) -> tuple[bool, list[str]]:
    path = Path(path)
    if not path.is_dir():
        return False, ["state folder"]
    patterns = {
        "model state": ("model*.safetensors", "pytorch_model*.bin"),
        "optimizer state": ("optimizer*.bin", "optimizer*.pt"),
        "scheduler state": ("scheduler*.bin", "scheduler*.pt"),
        "random-number state": ("random_states*.pkl",),
    }
    missing = []
    for label, globs in patterns.items():
        matches = [candidate for pattern in globs for candidate in path.glob(pattern)]
        if not any(_usable_file(candidate) for candidate in matches):
            missing.append(label)
    if exact_position and not state_epoch(path) and not state_step(path):
        missing.append("epoch/step position marker in the state-folder name")
    return not missing, missing


def _usable_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _job_settings(job: dict[str, Any]) -> dict[str, Any] | None:
    settings = job.get("settings_snapshot") or job.get("settings")
    return settings if isinstance(settings, dict) and settings else None


def effective_history_settings(job: dict[str, Any]) -> dict[str, Any]:
    """Apply compatibility rules to a historical job's effective settings."""

    snapshot = _job_settings(job)
    if not snapshot:
        raise ValueError("This job has no complete settings snapshot.")
    settings = copy.deepcopy(snapshot)
    is_minimax = str(settings.get("training_mode") or job.get("mode") or "").startswith("MiniMax H3")
    has_minimax_depth_schema = any(
        key in settings
        for key in (
            "minimax_h3_depth_vae_device",
            "minimax_h3_keep_depth_vae_on_device",
            "minimax_h3_depth_every_n_steps",
        )
    )
    recorded_commands = job.get("commands") or job.get("command") or []
    if isinstance(recorded_commands, str):
        command_text = recorded_commands
    else:
        command_text = " ".join(
            " ".join(str(item) for item in value)
            if isinstance(value, (list, tuple))
            else str(value)
            for value in recorded_commands
        )
    log_path = str(job.get("console_log_path") or "").strip()
    if log_path and not command_text:
        try:
            command_text = Path(log_path).read_text(encoding="utf-8", errors="replace")[:65536]
        except OSError:
            pass
    recorded_advanced_minimax = bool(
        re.search(r"(?:^|\s)--(?:depth_anchor_weight|weight_noise_sigma)(?:\s|=)", command_text)
    )
    if is_minimax and not has_minimax_depth_schema and not recorded_advanced_minimax:
        # These shared Krea values were inert before MiniMax depth support was
        # introduced. Do not silently activate them when an old job is replayed.
        settings["krea2_generalization_preset"] = "Off (Baseline)"
        settings["krea2_weight_noise_sigma"] = "0"
        settings["krea2_depth_anchor_weight"] = "0"
    return settings


def _recorded_final_stage_settings(job: dict[str, Any]) -> dict[str, Any] | None:
    """Return the effective recipe for the last completed staged run."""

    recorded = job.get("final_stage_settings")
    if isinstance(recorded, dict) and recorded:
        return recorded

    # Completed records written before final-stage lineage was introduced can
    # still be recovered deterministically from their saved stage plan.
    base = _job_settings(job)
    if (
        job.get("kind") != "staged_training"
        or job.get("status") != "completed"
        or not base
    ):
        return None
    stages = enabled_stages(base)
    if not stages:
        return None
    index = len(stages) - 1
    stage = stages[index]
    if stage.get("type", "standard") == "standard":
        try:
            return prepare_standard_stage(base, stage, index)
        except (KeyError, TypeError, ValueError):
            return None

    settings = copy.deepcopy(base)
    settings["stage_type"] = "face_refinement"
    settings["output_name"] = f"{base.get('output_name', 'training')}-{stage_label(stage, index)}"
    settings["max_train_steps"] = str(stage.get("steps") or "")
    settings["max_train_epochs"] = ""
    settings["resume_path"] = ""
    settings["resume_exact_position"] = False
    settings["recovery_mode"] = False
    return settings


def _recorded_final_stage_artifacts(job: dict[str, Any]) -> dict[str, str]:
    artifacts = job.get("final_stage_artifacts")
    if isinstance(artifacts, dict):
        return {
            "state": str(artifacts.get("state") or ""),
            "lora": str(artifacts.get("lora") or ""),
        }
    lineage = job.get("stage_lineage")
    if isinstance(lineage, list) and lineage:
        artifacts = lineage[-1].get("artifacts") if isinstance(lineage[-1], dict) else None
        if isinstance(artifacts, dict):
            return {
                "state": str(artifacts.get("state") or ""),
                "lora": str(artifacts.get("lora") or ""),
            }
    return {"state": "", "lora": ""}


def _add_settings_state_candidates(settings: dict[str, Any], add) -> None:
    try:
        for path in candidate_state_paths(settings):
            add(path)
        run_name = effective_run_name(settings)
        run_dir = Path(settings["output_dir"]) / run_name
        for path in run_dir.glob(f"{run_name}-*-state"):
            add(path)
    except (KeyError, TypeError, ValueError, OSError):
        pass


def state_candidates(job: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []

    def add(path: Path) -> None:
        try:
            if path.is_dir() and any(path.iterdir()) and path not in candidates:
                candidates.append(path)
        except OSError:
            pass

    final_settings = _recorded_final_stage_settings(job)
    if final_settings:
        final_state = _recorded_final_stage_artifacts(job)["state"].strip()
        if final_state:
            add(Path(final_state).expanduser())
        _add_settings_state_candidates(final_settings, add)

    settings = _job_settings(job)
    if settings and settings is not final_settings:
        _add_settings_state_candidates(settings, add)
    resume = str(job.get("resume_path") or "").strip()
    if resume:
        add(Path(resume).expanduser())
    return candidates


def resolve_continuation_state(job: dict[str, Any]) -> Path:
    final_settings = _recorded_final_stage_settings(job)
    if final_settings:
        candidates: list[Path] = []

        def add(path: Path) -> None:
            try:
                if path.is_dir() and any(path.iterdir()) and path not in candidates:
                    candidates.append(path)
            except OSError:
                pass

        final_state = _recorded_final_stage_artifacts(job)["state"].strip()
        if final_state:
            add(Path(final_state).expanduser())
        _add_settings_state_candidates(final_settings, add)
    else:
        candidates = state_candidates(job)
    if not candidates:
        target = " for its final completed stage" if final_settings else ""
        raise FileNotFoundError(f"No saved state folder could be found{target}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_continuation_lora(
    job: dict[str, Any],
    final_settings: dict[str, Any],
) -> Path:
    """Resolve adapter weights for an additive continuation.

    A normal run may have a standalone ``.safetensors`` export, while runs
    configured to save only Accelerate state keep the adapter model in
    ``*-state/model.safetensors``.  The latter is still a valid network-weight
    source for a fresh optimizer continuation; it must not be confused with
    ``resume_path`` (which is reserved for positional state recovery).
    """

    candidates = []

    def add(path: Path) -> None:
        if _usable_file(path) and path not in candidates:
            candidates.append(path)

    recorded = _recorded_final_stage_artifacts(job)["lora"].strip()
    if recorded:
        add(Path(recorded).expanduser())
    face_output = str(final_settings.get("face_output_path") or "").strip()
    if face_output:
        add(Path(face_output).expanduser())
    try:
        for path in candidate_lora_paths(final_settings):
            add(path)
    except (KeyError, TypeError, ValueError, OSError):
        pass
    # Prefer any checkpoint exports belonging to this run before looking at a
    # parent path recorded in ``resume_path``.  A continuation can have fewer
    # epochs than its parent, so a global newest-state sort would otherwise
    # accidentally select the parent's adapter.
    try:
        run_dir = Path(final_settings["output_dir"]).expanduser() / effective_run_name(final_settings)
        output_loras = [path for path in run_dir.glob("*.safetensors") if _usable_file(path)]
        output_loras.sort(key=lambda path: (state_epoch(path), state_step(path), path.stat().st_mtime), reverse=True)
        for path in output_loras:
            add(path)
        output_states = [path for path in run_dir.glob("*-state") if path.is_dir()]
        output_states.sort(key=lambda path: (state_epoch(path), state_step(path), path.stat().st_mtime), reverse=True)
        for state in output_states:
            add(state / "model.safetensors")
    except (KeyError, TypeError, ValueError, OSError):
        pass
    # Save-state-only runs do not always have a top-level LoRA export.  Their
    # model.safetensors contains the adapter weights; use the newest saved
    # checkpoint as an additive source instead of switching the UI to state
    # recovery and silently clearing the network-weight field.
    recorded_state = _recorded_final_stage_artifacts(job)["state"].strip()
    if recorded_state:
        add(Path(recorded_state).expanduser() / "model.safetensors")
    state_paths = state_candidates(job)
    state_paths.sort(
        key=lambda path: (state_epoch(path), state_step(path), path.stat().st_mtime),
        reverse=True,
    )
    for state in state_paths:
        add(state / "model.safetensors")
    if not candidates:
        raise FileNotFoundError(
            "No completed LoRA or saved adapter weights could be found for this job."
        )
    return candidates[0]


def resolve_exact_recovery_state(job: dict[str, Any]) -> tuple[Path | None, list[str]]:
    candidates = state_candidates(job)
    finished_timestamp = None
    try:
        finished_timestamp = datetime.fromisoformat(str(job.get("finished_at") or "")).timestamp()
    except ValueError:
        pass
    if finished_timestamp is not None:
        candidates = [path for path in candidates if path.stat().st_mtime <= finished_timestamp + 300]
    current_epoch = int(job.get("current_epoch") or 0)
    if current_epoch:
        bounded = [path for path in candidates if not state_epoch(path) or state_epoch(path) <= current_epoch]
        if bounded:
            candidates = bounded
    candidates.sort(key=lambda path: (state_step(path), state_epoch(path), path.stat().st_mtime), reverse=True)
    rejected = []
    for path in candidates:
        valid, missing = validate_accelerate_state(path, exact_position=True)
        if valid:
            return path, []
        rejected.append(f"{path.name}: missing {', '.join(missing)}")
    return None, rejected


def prepare_continuation(job: dict[str, Any]) -> dict[str, Any]:
    snapshot = effective_history_settings(job)
    final_settings = _recorded_final_stage_settings(job)
    settings = copy.deepcopy(snapshot)
    if final_settings:
        settings.update(copy.deepcopy(final_settings))
    try:
        lora = resolve_continuation_lora(job, final_settings or settings)
        settings["resume_path"] = ""
        settings["network_weights"] = str(lora)
        settings["starting_point_mode"] = "weights"
    except FileNotFoundError:
        # Keep a useful fallback for legacy output folders that contain only a
        # positional state directory.  New runs normally take the branch
        # above, so Continue as new is visibly a LoRA continuation whenever
        # adapter weights can be resolved.
        state = resolve_continuation_state(job)
        settings["resume_path"] = str(state)
        settings["network_weights"] = ""
        settings["starting_point_mode"] = "state"
    source_name = str(
        settings.get("output_name")
        or job.get("output_name")
        or job.get("name")
        or "training"
    )
    settings["output_name"] = continuation_name(source_name)
    settings["save_state"] = True
    settings["recache_latents"] = False
    settings["recache_text"] = False
    settings["use_staged_training"] = False
    settings["resume_exact_position"] = False
    settings["recovery_mode"] = False
    settings.pop("stage_type", None)
    settings.pop("face_output_path", None)
    return settings


def prepare_exact_recovery(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("kind") != "training" or job.get("status") not in {"failed", "stopped", "completed"}:
        raise ValueError(
            "Exact resume is available only for completed, failed, or stopped normal training jobs."
        )
    snapshot = effective_history_settings(job)
    state, rejected = resolve_exact_recovery_state(job)
    if state is None:
        detail = "; ".join(rejected[:4]) or "No associated state folder was found."
        raise ValueError(f"No complete positional Accelerate state is available. {detail}")
    settings = copy.deepcopy(snapshot)
    settings["output_name"] = str(settings.get("output_name") or job.get("output_name") or "training")
    settings["resume_path"] = str(state)
    settings["network_weights"] = ""
    settings["starting_point_mode"] = "state"
    settings["save_state"] = True
    settings["recache_latents"] = False
    settings["recache_text"] = False
    settings["use_staged_training"] = False
    settings["resume_exact_position"] = True
    settings["recovery_mode"] = True
    return settings


def prepare_face_refinement(job: dict[str, Any]) -> dict[str, Any]:
    """Load a recorded Krea run as a one-stage face-refinement recipe."""
    snapshot = job.get("settings_snapshot") or job.get("settings")
    if not isinstance(snapshot, dict) or snapshot.get("training_mode") != "Krea 2":
        raise ValueError("Face refinement needs a recorded Krea 2 job with a complete settings snapshot.")
    from musubi_tuner.face_refinement.face_models import default_model_dir
    from musubi_tuner.face_refinement.lora_validation import validate_krea2_lora
    from musubi_tuner.face_refinement.pose_plan import default_pose_plan

    run_name = effective_run_name(snapshot)
    run_dir = Path(str(snapshot.get("output_dir") or "")).expanduser() / run_name
    candidates = []
    face_output = str(snapshot.get("face_output_path") or "").strip()
    if face_output:
        candidates.append(Path(face_output))
    epoch_text = str(snapshot.get("max_train_epochs") or "").strip()
    if epoch_text.isdigit():
        candidates.append(run_dir / f"{run_name}-{int(epoch_text):06d}.safetensors")
    candidates.append(run_dir / f"{run_name}.safetensors")
    candidates.extend(path / "model.safetensors" for path in state_candidates(job))
    input_lora = None
    for candidate in candidates:
        try:
            validate_krea2_lora(candidate.resolve())
            input_lora = candidate.resolve()
            break
        except (OSError, ValueError):
            continue
    if input_lora is None:
        raise FileNotFoundError("No complete Krea 2 LoRA could be found for this job.")

    settings = copy.deepcopy(snapshot)
    source_name = str(settings.get("output_name") or job.get("output_name") or "krea-lora")
    candidate_name = f"{source_name}-face"
    suffix = 2
    while (Path(str(settings.get("output_dir") or "")) / candidate_name).exists():
        candidate_name = f"{source_name}-face-{suffix}"
        suffix += 1
    defaults = {
        "input_mode": "existing_lora", "input_lora": str(input_lora), "trigger_word": "",
        "excluded_reference_images": [], "reference_dir": "", "face_model_dir": str(default_model_dir()),
        "prompts": ["portrait photo of {trigger}, natural expression, soft daylight"],
        "steps": 30, "resolution": 512, "denoise_steps": 12, "draft_k": 1, "cfg_scale": 5.5,
        "learning_rate": 1e-4, "target_similarity": 0.45, "stop_similarity": 0.55,
        "early_stop_patience": 5, "min_detection_rate": 0.25, "anti_copy_weight": 0.02,
        "preview_every": 5, "save_every": 10, "qkvo_only": True, "checkpoint_vae": True,
        "license_acknowledged": False, "pose_aware": False, "pose_reward_weight": 0.20,
        "pose_min_references": 2, "pose_plan": default_pose_plan(), "blocks_to_swap": 10, "gpu_id": "auto",
    }
    defaults.update(copy.deepcopy(snapshot.get("face_refinement_config") or {}))
    defaults.update({"input_mode": "existing_lora", "input_lora": str(input_lora)})
    settings.update({
        "output_name": candidate_name, "resume_path": "", "network_weights": "",
        "use_staged_training": True, "resume_exact_position": False, "recovery_mode": False,
        "staged_training_config": [{
            "label": "face-refinement", "enabled": True, "type": "face_refinement",
            "dataset_config": "", "epochs": "", "steps": str(defaults["steps"]),
        }],
        "face_refinement_config": defaults,
    })
    return settings
