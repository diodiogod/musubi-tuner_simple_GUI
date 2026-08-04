from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from backends import flux2, krea2, minimax_h3, wan
from modern_gui.sample_prompts import prepare_sample_prompt_settings
from modern_gui.training_notes import effective_training_comment


def _accelerate_executable() -> str:
    """Resolve Accelerate from the same environment as the modern GUI server."""

    executable_name = "accelerate.exe" if os.name == "nt" else "accelerate"
    environment_candidate = Path(sys.executable).resolve().with_name(executable_name)
    if environment_candidate.is_file():
        return str(environment_candidate)
    discovered = shutil.which("accelerate")
    if discovered:
        return discovered
    raise FileNotFoundError(
        f"Could not find {executable_name} beside the GUI Python ({sys.executable}) or on PATH. "
        "Install Accelerate in the GUI environment before starting training."
    )


def _resolve_training_launchers(commands: list[list[str]]) -> list[list[str]]:
    accelerate = None
    resolved = []
    for command in commands:
        command = list(command)
        if command and str(command[0]).lower() == "accelerate":
            accelerate = accelerate or _accelerate_executable()
            command[0] = accelerate
        resolved.append(command)
    return resolved


def create_wan_i2v_cache_config(original_config_path: str) -> str:
    """Create the cache-only Wan I2V TOML used by the classic GUI.

    The canonical dataset document remains untouched. Only the execution copy
    passed to Musubi's cache scripts omits ``image_directory`` entries.
    """

    source = Path(original_config_path).expanduser()
    content = source.read_text(encoding="utf-8-sig")
    filtered = "\n".join(
        line
        for line in content.splitlines()
        if not line.lstrip().startswith("image_directory")
    )
    handle, temporary_name = tempfile.mkstemp(
        prefix=f"{source.stem or 'dataset'}-wan-i2v-cache-",
        suffix=".toml",
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(filtered)
            if content.endswith(("\n", "\r")):
                stream.write("\n")
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return temporary_name


def build_command_plan(settings: dict[str, Any], preview: bool = False) -> dict[str, list[list[str]]]:
    settings = dict(settings)
    starting_point = settings.get("starting_point_mode")
    if starting_point == "new":
        settings["network_weights"] = ""
        settings["resume_path"] = ""
        settings["resume_exact_position"] = False
        settings["recovery_mode"] = False
    elif starting_point == "weights":
        settings["resume_path"] = ""
        settings["resume_exact_position"] = False
        settings["recovery_mode"] = False
    elif starting_point == "state":
        settings["network_weights"] = ""
    settings = prepare_sample_prompt_settings(settings, write=not preview)
    settings["training_comment"] = effective_training_comment(settings)
    if preview:
        settings["_preview_only"] = True
    mode = settings.get("training_mode", "Wan 2.2")
    python_executable = sys.executable
    if mode == "Wan 2.2":
        cache_config_by_source: dict[str, str] = {}

        def cache_config(source: str) -> str:
            key = str(Path(source).expanduser().resolve())
            if key not in cache_config_by_source:
                cache_config_by_source[key] = create_wan_i2v_cache_config(source)
            return cache_config_by_source[key]

        cache = wan.build_cache_commands(
            settings,
            python_executable,
            temp_config_fn=cache_config if not preview else None,
        )
        train = wan.build_commands(settings)
    elif mode == "Krea 2":
        cache = krea2.build_cache_commands(settings, python_executable)
        train = krea2.build_commands(settings)
    elif mode == "MiniMax H3 (Experimental)":
        cache = minimax_h3.build_cache_commands(settings, python_executable)
        train = minimax_h3.build_commands(settings)
    elif mode in {"Flux.2 Klein", "Flux.2 Dev"}:
        cache = flux2.build_cache_commands(settings, python_executable)
        train = flux2.build_commands(settings)
    else:
        raise ValueError(f"Unsupported training mode: {mode}")
    return {"cache": cache, "train": _resolve_training_launchers(train)}
