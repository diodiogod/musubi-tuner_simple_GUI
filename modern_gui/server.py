from __future__ import annotations

import argparse
import ipaddress
import json
import mimetypes
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.parse import parse_qs

from modern_gui.commands import build_command_plan
from modern_gui.dataset_documents import (
    DocumentConflictError,
    add_dataset,
    add_datasets,
    duplicate_dataset,
    inspect_dataset_sources,
    load_document,
    move_dataset,
    remove_dataset,
    save_document,
    split_dataset_subfolders,
    summarize_document,
    toggle_dataset_disabled,
    update_dataset,
    update_general,
)
from modern_gui.dataset_media import (
    CaptionConflictError,
    MediaTokenError,
    dataset_source_location,
    list_dataset_media,
    resolve_media_token,
    save_media_caption,
)
from modern_gui.h3_datasets import (
    build_image_audio_videos,
    inspect_pairing_sources,
    load_ref2va_document,
    pairing_plan,
    save_ref2va_document,
)
from modern_gui.jobs import SUPERVISOR
from job_performance import enrich_job, read_job_log
from modern_gui.recovery import (
    clear_desktop_history,
    effective_history_settings,
    import_output_jobs,
    load_desktop_history,
    prepare_continuation,
    prepare_exact_recovery,
    prepare_face_refinement,
)
from modern_gui.monitor import gpu_snapshot
from modern_gui.samples import (
    add_sample_source,
    allowed_sample_roots,
    discover_samples,
    find_nearby_sample_sources,
    load_sample_sources,
    remove_sample_source,
    resolve_sample_file,
    sample_source_status,
)
from modern_gui.sampling import estimate_steps_per_epoch
from modern_gui.validation import require_valid_training_settings, validate_training_settings
from modern_gui.settings import load_settings, save_settings, settings_schema


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
LOCAL_ACTION_POST_PATHS = frozenset(
    {
        "/api/settings",
        "/api/dataset/save",
        "/api/dataset/toggle-disabled",
        "/api/dataset/media",
        "/api/dataset/open-source",
        "/api/dataset/caption",
        "/api/h3/pairing/inspect",
        "/api/h3/pairing/build",
        "/api/h3/ref2va/load",
        "/api/h3/ref2va/save",
        "/api/face/preflight",
        "/api/face/models/download",
        "/api/prompt-library/import",
        "/api/prompt-library/update",
        "/api/prompt-library/favorite",
        "/api/prompt-library/delete",
        "/api/prompts/preview",
        "/api/path/select",
        "/api/path/drop",
        "/api/estimate-lora",
        "/api/workspace/apply",
        "/api/face/evaluate",
        "/api/face/open-results",
        "/api/tools/convert",
        "/api/tools/accelerate-config",
        "/api/jobs/start",
        "/api/jobs/stop",
        "/api/jobs/stop-after-next-epoch",
        "/api/jobs/open-path",
        "/api/jobs/import-found",
        "/api/jobs/clear",
        "/api/legacy/start",
        "/api/samples/sources",
    }
)


class MusubiWebHandler(BaseHTTPRequestHandler):
    server_version = "MusubiModernGUI/0.1"

    def _json(self, payload, status=HTTPStatus.OK):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 8 * 1024 * 1024:
            raise ValueError("Request body is too large.")
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def _error(self, exc, status=HTTPStatus.BAD_REQUEST):
        self._json({"error": str(exc)}, status)

    def _require_loopback_write(self):
        try:
            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                raise PermissionError("State-changing actions are only allowed from this computer.")
        except ValueError as exc:
            raise PermissionError("Could not verify the local client address.") from exc
        origin = self.headers.get("Origin")
        if origin:
            parsed_origin = urlparse(origin)
            hostname = parsed_origin.hostname
            request_host = str(self.headers.get("Host") or "").casefold()
            if (
                parsed_origin.scheme != "http"
                or hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed_origin.netloc.casefold() != request_host
            ):
                raise PermissionError("State-changing actions require the same local application origin.")

    def _send_file(self, path: Path, *, allow_ranges: bool = False):
        requested = path.resolve()
        size = requested.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range") if allow_ranges else None
        if range_header:
            if size == 0:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", "bytes */0")
                self.end_headers()
                return
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            elif last:
                suffix = int(last)
                start = max(0, size - suffix)
                end = size - 1
            if start < 0 or start >= size or end < start:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = size if size == 0 else max(0, end - start + 1)
        content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        if requested.suffix.lower() == ".jxl":
            content_type = "image/jxl"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "private, max-age=30")
        if allow_ranges:
            self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with requested.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                return self._json({"ok": True, "root": str(ROOT)})
            if parsed.path == "/api/settings":
                settings = load_settings()
                return self._json({"settings": settings, "schema": settings_schema(settings)})
            if parsed.path == "/api/settings/defaults":
                defaults_path = ROOT / "Base_SETTINGS.json"
                defaults = json.loads(defaults_path.read_text(encoding="utf-8")) if defaults_path.is_file() else {}
                return self._json({"settings": defaults})
            if parsed.path == "/api/face/defaults":
                from musubi_tuner.face_refinement.face_models import default_model_dir
                from musubi_tuner.face_refinement.pose_plan import default_pose_plan

                return self._json(
                    {"face_model_dir": str(default_model_dir()), "pose_plan": default_pose_plan()}
                )
            if parsed.path == "/api/dataset":
                query = parsed.query.removeprefix("path=")
                return self._json(load_document(unquote(query)))
            if parsed.path == "/api/dataset/media-file":
                token = parse_qs(parsed.query).get("token", [""])[0]
                access = resolve_media_token(token)
                return self._send_file(access.path, allow_ranges=access.kind == "video")
            if parsed.path == "/api/jobs":
                modern = [enrich_job(dict(job, _source="web", _history_index=index)) for index, job in enumerate(SUPERVISOR.history())]
                desktop = [enrich_job(job) for job in load_desktop_history()]
                jobs = sorted(modern + desktop, key=lambda job: str(job.get("started_at") or ""), reverse=True)
                return self._json({"jobs": jobs})
            if parsed.path == "/api/jobs/active":
                after = 0
                if parsed.query.startswith("after="):
                    after = int(parsed.query.removeprefix("after="))
                return self._json(SUPERVISOR.snapshot(after))
            if parsed.path == "/api/jobs/log":
                query = parse_qs(parsed.query)
                source = query.get("source", ["desktop"])[0]
                index = int(query.get("index", ["-1"])[0])
                jobs = load_desktop_history() if source == "desktop" else SUPERVISOR.history()
                if index < 0 or index >= len(jobs):
                    raise IndexError("The selected history entry no longer exists.")
                job = jobs[index]
                return self._json(
                    {
                        "log": read_job_log(job.get("console_log_path") or ""),
                        "performance": enrich_job(job)["performance"],
                    }
                )
            if parsed.path == "/api/gpu":
                return self._json(gpu_snapshot())
            if parsed.path == "/api/samples":
                query = parse_qs(parsed.query)
                output_dir = unquote(query.get("output_dir", [""])[0])
                output_name = unquote(query.get("output_name", [""])[0])
                return self._json(discover_samples(output_dir, output_name, source_paths=load_sample_sources()))
            if parsed.path == "/api/samples/sources":
                return self._json({"sources": sample_source_status()})
            if parsed.path == "/api/samples/sources/nearby":
                query = parse_qs(parsed.query)
                return self._json(
                    {
                        "sources": find_nearby_sample_sources(
                            unquote(query.get("output_dir", [""])[0]),
                            unquote(query.get("output_name", [""])[0]),
                        )
                    }
                )
            if parsed.path == "/api/prompt-library":
                from prompt_library import PromptLibraryStore

                store = PromptLibraryStore()
                return self._json({"prompts": store.prompts, "root": str(store.root)})
            if parsed.path == "/api/prompt-library/thumbnail":
                from prompt_library import PromptLibraryStore

                store = PromptLibraryStore()
                entry_id = parse_qs(parsed.query).get("id", [""])[0]
                entry = store.find(library_id=entry_id)
                thumbnail = store.latest_thumbnail(entry) if entry else None
                if not thumbnail:
                    raise FileNotFoundError("This library prompt has no test thumbnail.")
                requested = store.thumbnail_path(thumbnail).resolve()
                library_root = store.root.resolve()
                if not requested.is_file() or library_root not in requested.parents:
                    raise PermissionError("Prompt thumbnail is outside the library folder.")
                content = requested.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(requested.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "private, max-age=30")
                self.end_headers()
                self.wfile.write(content)
                return
            if parsed.path == "/api/sample-file":
                query = parse_qs(parsed.query)
                requested = unquote(query.get("path", [""])[0])
                settings = load_settings()
                history = SUPERVISOR.history() + load_desktop_history()
                sample, _content_type = resolve_sample_file(requested, allowed_sample_roots(settings, history))
                return self._send_file(sample, allow_ranges=sample.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"})
            if parsed.path == "/api/face-image":
                query = parse_qs(parsed.query)
                requested = Path(unquote(query.get("path", [""])[0])).expanduser().resolve()
                config = load_settings().get("face_refinement_config") or {}
                reference_root = Path(str(config.get("reference_dir") or "")).expanduser().resolve()
                if not requested.is_file() or reference_root not in requested.parents:
                    raise PermissionError("Face reference is outside the configured reference folder.")
                content = requested.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(requested.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "private, max-age=30")
                self.end_headers()
                self.wfile.write(content)
                return
            if parsed.path == "/api/evaluation-image":
                query = parse_qs(parsed.query)
                requested = Path(unquote(query.get("path", [""])[0])).expanduser().resolve()
                output_root = Path(str(load_settings().get("output_dir") or "")).expanduser().resolve()
                if not requested.is_file() or output_root not in requested.parents:
                    raise PermissionError("Evaluation image is outside the configured output folder.")
                content = requested.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(requested.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "private, max-age=30")
                self.end_headers()
                self.wfile.write(content)
                return
            self._static(parsed.path)
        except (MediaTokenError, PermissionError) as exc:
            self._error(exc, HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            self._error(exc, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._error(exc)

    def do_POST(self):
        try:
            if self.path in LOCAL_ACTION_POST_PATHS:
                self._require_loopback_write()
            body = self._body()
            if self.path == "/api/settings":
                return self._json({"settings": save_settings(body.get("settings", {}))})
            if self.path == "/api/samples/sources":
                if body.get("remove"):
                    sources = remove_sample_source(body.get("path", ""))
                else:
                    sources = add_sample_source(body.get("path", ""), body.get("label", ""))
                return self._json({"sources": sample_source_status()})
            if self.path == "/api/commands/preview":
                settings = body.get("settings", {})
                return self._json(
                    {
                        **build_command_plan(settings, preview=True),
                        "validation": validate_training_settings(settings),
                    }
                )
            if self.path == "/api/settings/validate":
                return self._json(validate_training_settings(body.get("settings", {})))
            if self.path == "/api/dataset/parse":
                return self._json(summarize_document(body.get("text", ""), body.get("path", "")))
            if self.path == "/api/dataset/save":
                return self._json(
                    save_document(
                        body.get("path", ""),
                        body.get("text", ""),
                        body.get("expected_revision"),
                    )
                )
            if self.path == "/api/dataset/general":
                return self._json(
                    update_general(
                        body.get("text", ""),
                        body.get("changes", {}),
                        body.get("path", ""),
                    )
                )
            if self.path == "/api/dataset/update":
                return self._json(
                    update_dataset(
                        body.get("text", ""),
                        int(body.get("index", -1)),
                        body.get("changes", {}),
                        body.get("path", ""),
                    )
                )
            if self.path == "/api/dataset/add":
                common = {
                    "text": body.get("text", ""),
                    "kind": body.get("kind", "image"),
                    "source_path": body.get("path", ""),
                    "architecture": body.get("architecture", ""),
                }
                source_paths = body.get("source_paths")
                if isinstance(source_paths, list):
                    return self._json(add_datasets(folder_paths=source_paths, **common))
                return self._json(add_dataset(folder_path=body.get("source_path", ""), **common))
            if self.path == "/api/dataset/split-subfolders":
                return self._json(
                    split_dataset_subfolders(
                        body.get("text", ""),
                        int(body.get("index", -1)),
                        body.get("path", ""),
                    )
                )
            if self.path == "/api/dataset/toggle-disabled":
                return self._json(
                    toggle_dataset_disabled(
                        body.get("text", ""),
                        int(body.get("index", -1)),
                        bool(body.get("disabled", True)),
                        body.get("path", ""),
                    )
                )
            if self.path == "/api/dataset/remove":
                return self._json(
                    remove_dataset(body.get("text", ""), int(body.get("index", -1)), body.get("path", ""))
                )
            if self.path == "/api/dataset/duplicate":
                return self._json(
                    duplicate_dataset(body.get("text", ""), int(body.get("index", -1)), body.get("path", ""))
                )
            if self.path == "/api/dataset/move":
                return self._json(
                    move_dataset(
                        body.get("text", ""),
                        int(body.get("index", -1)),
                        int(body.get("destination", -1)),
                        body.get("path", ""),
                    )
                )
            if self.path == "/api/dataset/inspect":
                return self._json(inspect_dataset_sources(body.get("text", ""), body.get("path", "")))
            if self.path == "/api/h3/pairing/inspect":
                inspection = inspect_pairing_sources(body.get("image_directory", ""), body.get("audio_directory", ""))
                inspection["plan"] = pairing_plan(
                    body.get("image_directory", ""),
                    body.get("audio_directory", ""),
                    strategy=body.get("strategy", "round_robin"),
                    seed=int(body.get("seed", 42)),
                )
                return self._json(inspection)
            if self.path == "/api/h3/pairing/build":
                return self._json(
                    build_image_audio_videos(
                        body.get("image_directory", ""),
                        body.get("audio_directory", ""),
                        body.get("output_directory", ""),
                        strategy=body.get("strategy", "round_robin"),
                        seed=int(body.get("seed", 42)),
                        width=int(body.get("width", 768)),
                        height=int(body.get("height", 768)),
                        target_frames=int(body.get("target_frames", 124)),
                        allow_experimental_duration=bool(body.get("allow_experimental_duration", False)),
                    )
                )
            if self.path == "/api/h3/ref2va/load":
                return self._json(load_ref2va_document(body.get("path", "")))
            if self.path == "/api/h3/ref2va/save":
                return self._json(save_ref2va_document(body.get("path", ""), body.get("records", [])))
            if self.path == "/api/dataset/estimate-steps":
                dataset_path = str(body.get("path", "")).strip()
                if not dataset_path:
                    raise ValueError("Choose a dataset TOML before estimating epoch steps.")
                return self._json(
                    estimate_steps_per_epoch(
                        dataset_path,
                        body.get("gradient_accumulation_steps", 1),
                        body.get("text") or None,
                    )
                )
            if self.path == "/api/dataset/media":
                return self._json(
                    list_dataset_media(
                        body.get("text", ""),
                        body.get("path", ""),
                        int(body.get("index", 0)),
                        int(body.get("page", 1)),
                        int(body.get("page_size", 24)),
                        str(body.get("query", "")),
                        str(body.get("filter", "all")),
                    )
                )
            if self.path == "/api/dataset/open-source":
                target = dataset_source_location(
                    body.get("text", ""),
                    int(body.get("index", 0)),
                )
                if sys.platform == "win32":
                    os.startfile(target)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)])
                else:
                    subprocess.Popen(["xdg-open", str(target)])
                return self._json({"opened": str(target)})
            if self.path == "/api/dataset/caption":
                return self._json(
                    save_media_caption(
                        str(body.get("token", "")),
                        str(body.get("caption", "")),
                        str(body.get("expected_revision", "")),
                    )
                )
            if self.path == "/api/face/preflight":
                from musubi_tuner.face_refinement.preflight import scan_reference_faces

                return self._json(
                    scan_reference_faces(
                        str(body.get("reference_dir", "")),
                        str(body.get("face_model_dir", "")),
                    )
                )
            if self.path == "/api/face/models/download":
                from musubi_tuner.face_refinement.face_models import ensure_models

                model_dir = str(body.get("face_model_dir", "")).strip()
                if not model_dir:
                    raise ValueError("Choose a face model directory first.")
                return self._json({"model_dir": str(ensure_models(model_dir))})
            if self.path == "/api/face/pose-preset":
                from copy import deepcopy
                from musubi_tuner.face_refinement.pose_plan import apply_preset, default_pose_plan

                preset = str(body.get("preset", "balanced_identity"))
                if preset not in {"balanced_identity", "improve_profiles", "improve_three_quarter"}:
                    raise ValueError("Unknown pose-plan preset.")
                plan = deepcopy(body.get("plan") or default_pose_plan(preset))
                plan["enabled"] = True
                return self._json({"pose_plan": apply_preset(plan, preset)})
            if self.path == "/api/face/weak-plan":
                from copy import deepcopy
                from musubi_tuner.face_refinement.pose_plan import (
                    TRAINABLE_POSES,
                    default_pose_plan,
                    normalize_pose_plan,
                )

                result = body.get("result") or {}
                plan = deepcopy(body.get("plan") or default_pose_plan("custom"))
                plan["enabled"] = True
                plan["preset"] = "custom"
                target = float(body.get("target_similarity", 0.55))
                for pose in TRAINABLE_POSES:
                    metrics = (result.get("poses") or {}).get(pose) or {}
                    bucket = plan.setdefault("buckets", {}).setdefault(pose, {})
                    samples = int(metrics.get("samples") or 0)
                    identity = metrics.get("pose_similarity")
                    if identity is None:
                        identity = metrics.get("overall_similarity")
                    if not samples or identity is None:
                        bucket.update({"enabled": False, "share": 0.0})
                        continue
                    pose_success = float(metrics.get("pose_success_rate") or 0.0)
                    weakness = max(0.0, target - float(identity)) + 0.20 * (1.0 - pose_success)
                    bucket.update({"enabled": weakness > 0.01, "share": weakness * 100.0})
                normalized, warnings = normalize_pose_plan(
                    plan,
                    body.get("reference_counts") or {},
                    int(body.get("min_references") or 2),
                )
                return self._json({"pose_plan": normalized, "warnings": warnings})
            if self.path == "/api/prompt-library/import":
                from prompt_library import PromptLibraryStore

                added, merged = PromptLibraryStore().import_prompts(
                    body.get("prompts", []),
                    source={"type": "modern_gui"},
                )
                return self._json({"added": added, "merged": merged})
            if self.path == "/api/prompt-library/update":
                from prompt_library import PromptLibraryStore

                store = PromptLibraryStore()
                store.update_entry(
                    str(body.get("id", "")),
                    name=str(body.get("name", "")),
                    prompt_data=body.get("prompt_data") or {},
                    tags=body.get("tags") or [],
                    collection=str(body.get("collection", "")),
                )
                return self._json({"updated": True})
            if self.path == "/api/prompt-library/favorite":
                from prompt_library import PromptLibraryStore

                store = PromptLibraryStore()
                entry_id = str(body.get("id", ""))
                if not store.find(library_id=entry_id):
                    raise KeyError("The library prompt no longer exists.")
                store.toggle_favorite(entry_id)
                return self._json({"updated": True})
            if self.path == "/api/prompt-library/delete":
                from prompt_library import PromptLibraryStore

                if not PromptLibraryStore().delete(str(body.get("id", ""))):
                    raise KeyError("The library prompt no longer exists.")
                return self._json({"deleted": True})
            if self.path == "/api/prompts/preview":
                from modern_gui.prompt_preview import build_prompt_preview

                active = SUPERVISOR.snapshot(after=2**63 - 1).get("active")
                if active and active.get("status") in {"starting", "running", "stopping"}:
                    raise RuntimeError("A web GUI job is already active.")
                settings = dict(body.get("settings", {}))
                prompts = [item for item in body.get("prompts", []) if item.get("enabled", True)]
                command, save_path = build_prompt_preview(settings, prompts)
                prompt_file = (
                    Path(command[command.index("--from_file") + 1])
                    if "--from_file" in command
                    else None
                )
                resolved_lora = (
                    str(command[command.index("--lora_weight") + 1])
                    if "--lora_weight" in command
                    else str(command[command.index("--network_weights") + 1])
                    if "--network_weights" in command
                    else ""
                )
                resolved_lora_multiplier = (
                    str(command[command.index("--lora_multiplier") + 1])
                    if "--lora_multiplier" in command
                    else ""
                )
                is_h3 = settings.get("training_mode") == "MiniMax H3 (Experimental)"
                preview_mode = "MiniMax H3 (Experimental)" if is_h3 else ("Krea 2 Turbo" if "--turbo" in command else "Krea 2")
                try:
                    existing_outputs = [
                        str(path.resolve()) for path in save_path.glob("*")
                        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov", ".m4v"}
                    ]
                    job = SUPERVISOR.start_commands(
                        [command],
                        name=f"{settings.get('output_name') or preview_mode} sample preview",
                        mode=preview_mode,
                        kind="sample_test",
                        settings=settings,
                        completion_context={
                            "kind": "sample_test", "save_path": str(save_path),
                            "existing_outputs": existing_outputs, "prompts": prompts, "mode": preview_mode,
                            "output_name": settings.get("output_name", ""),
                            "network_weights": resolved_lora,
                        },
                    )
                except Exception:
                    if prompt_file is not None:
                        prompt_file.unlink(missing_ok=True)
                    raise
                return self._json(
                    {
                        "job": job,
                        "save_path": str(save_path),
                        "preview_mode": preview_mode,
                        "network_weights": resolved_lora,
                        "lora_multiplier": resolved_lora_multiplier,
                    }
                )
            if self.path == "/api/path/select":
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                try:
                    if body.get("kind") == "directory":
                        selected = filedialog.askdirectory(initialdir=body.get("initial") or None, parent=root)
                    else:
                        selected = filedialog.askopenfilename(initialdir=body.get("initial") or None, parent=root)
                finally:
                    root.destroy()
                return self._json({"path": selected or ""})
            if self.path == "/api/path/drop":
                from modern_gui.path_drop import choose_directories

                selected = choose_directories(body.get("title") or "Drop dataset folders")
                return self._json({"paths": selected, "path": selected[0] if selected else ""})
            if self.path == "/api/estimate-lora":
                from musubi_tuner_gui import MusubiTunerGUI

                estimator = MusubiTunerGUI.__new__(MusubiTunerGUI)
                estimator._lora_shape_cache = {}
                byte_count, layer_count = estimator._estimate_adapter_bytes(
                    str(body.get("model_path", "")),
                    str(body.get("mode", "")),
                    int(body.get("rank") or 0),
                    str(body.get("network_type", "LoRA")),
                    int(body.get("lokr_factor") or -1),
                )
                return self._json(
                    {
                        "bytes": byte_count,
                        "layers": layer_count,
                        "formatted": MusubiTunerGUI._format_estimated_size(byte_count),
                    }
                )
            if self.path == "/api/workspace/apply":
                root = Path(str(body.get("root", ""))).expanduser()
                if not str(body.get("root", "")).strip():
                    raise ValueError("Choose a concept workspace folder first.")
                models = root / "models"
                logs = root / "log"
                models.mkdir(parents=True, exist_ok=True)
                logs.mkdir(parents=True, exist_ok=True)
                return self._json(
                    {"output_dir": str(models.resolve()), "logging_dir": str(logs.resolve())}
                )
            if self.path == "/api/face/evaluate":
                from backends.krea2_face_eval import prepare

                settings = dict(body.get("settings", {}))
                if settings.get("training_mode") != "Krea 2":
                    raise ValueError(
                        "Fixed Turbo face evaluation is Krea 2-only. MiniMax H3 face refinement is available, "
                        "but its fixed evaluation recipe has not been validated yet."
                    )
                config = dict(settings.get("face_refinement_config") or {})
                input_lora = str(body.get("input_lora") or config.get("input_lora") or "")
                baseline = body.get("baseline_result") or None
                prepared = prepare(
                    settings,
                    config,
                    input_lora,
                    baseline_result=baseline,
                    label="comparison" if baseline else "baseline",
                )
                job = SUPERVISOR.start_commands(
                    prepared["commands"],
                    name=f"{settings.get('output_name') or 'Krea 2'} face evaluation",
                    mode="Krea 2",
                    kind="face_evaluation",
                    settings=settings,
                )
                return self._json(
                    {"job": job, "result": str(prepared["result"]), "cases": prepared["cases"]}
                )
            if self.path == "/api/face/result":
                path = Path(str(body.get("path", ""))).expanduser().resolve()
                settings = load_settings()
                output_root = Path(str(settings.get("output_dir") or "")).expanduser().resolve()
                if not path.is_file() or output_root not in path.parents:
                    raise FileNotFoundError("The evaluation result is missing or outside the configured output folder.")
                return self._json(json.loads(path.read_text(encoding="utf-8")))
            if self.path == "/api/face/open-results":
                path = Path(str(body.get("path", ""))).expanduser().resolve()
                output_root = Path(str(load_settings().get("output_dir") or "")).expanduser().resolve()
                target = path.parent if path.is_file() else path
                if not target.is_dir() or output_root not in target.parents:
                    raise FileNotFoundError("The evaluation folder is missing or outside the configured output folder.")
                if sys.platform == "win32":
                    os.startfile(target)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)])
                else:
                    subprocess.Popen(["xdg-open", str(target)])
                return self._json({"opened": str(target)})
            if self.path == "/api/tools/convert":
                lora = Path(str(body.get("input", ""))).expanduser()
                output_dir = Path(str(body.get("output_dir", ""))).expanduser()
                if not lora.is_file() or not output_dir.is_dir():
                    raise ValueError("Choose an existing LoRA and output directory.")
                output = output_dir / f"{lora.stem}_converted.safetensors"
                command = [
                    sys.executable, "src/musubi_tuner/convert_lora.py",
                    "--input", str(lora), "--output", str(output),
                    "--target", str(body.get("target", "default")),
                ]
                job = SUPERVISOR.start_commands(
                    [command], name=output.name, mode="Utility", kind="conversion",
                    settings={"output_dir": str(output_dir), "output_name": output.name},
                )
                return self._json({"job": job, "output": str(output)})
            if self.path == "/api/tools/accelerate-config":
                accelerate = Path(sys.executable).parent / ("accelerate.exe" if sys.platform == "win32" else "accelerate")
                command = f'"{accelerate if accelerate.exists() else "accelerate"}" config'
                if sys.platform == "win32":
                    process = subprocess.Popen(f"start cmd /k {command}", shell=True)
                elif sys.platform == "darwin":
                    process = subprocess.Popen(["osascript", "-e", f'tell application "Terminal" to do script "{command}"'])
                else:
                    process = subprocess.Popen(["x-terminal-emulator", "-e", command])
                return self._json({"started": True, "pid": process.pid})
            if self.path == "/api/jobs/start":
                settings = body.get("settings", {})
                require_valid_training_settings(settings)
                return self._json({"job": SUPERVISOR.start(settings, bool(body.get("run_cache", True)))})
            if self.path == "/api/jobs/stop":
                return self._json({"job": SUPERVISOR.stop()})
            if self.path == "/api/jobs/stop-after-next-epoch":
                return self._json({"job": SUPERVISOR.stop_after_next_epoch(bool(body.get("enabled", True)))})
            if self.path in {"/api/jobs/prepare-continuation", "/api/jobs/prepare-recovery"}:
                source = body.get("source", "desktop")
                index = int(body.get("index", -1))
                jobs = load_desktop_history() if source == "desktop" else SUPERVISOR.history()
                if index < 0 or index >= len(jobs):
                    raise IndexError("The selected history entry no longer exists.")
                prepare = prepare_continuation if self.path.endswith("continuation") else prepare_exact_recovery
                return self._json({"settings": prepare(jobs[index])})
            if self.path == "/api/jobs/replay-settings":
                source = body.get("source", "desktop")
                index = int(body.get("index", -1))
                jobs = load_desktop_history() if source == "desktop" else SUPERVISOR.history()
                if index < 0 or index >= len(jobs):
                    raise IndexError("The selected history entry no longer exists.")
                return self._json({"settings": effective_history_settings(jobs[index])})
            if self.path == "/api/jobs/prepare-face":
                source = body.get("source", "desktop")
                index = int(body.get("index", -1))
                jobs = load_desktop_history() if source == "desktop" else SUPERVISOR.history()
                if index < 0 or index >= len(jobs):
                    raise IndexError("The selected history entry no longer exists.")
                return self._json({"settings": prepare_face_refinement(jobs[index])})
            if self.path == "/api/jobs/open-path":
                source = body.get("source", "desktop")
                index = int(body.get("index", -1))
                jobs = load_desktop_history() if source == "desktop" else SUPERVISOR.history()
                if index < 0 or index >= len(jobs):
                    raise IndexError("The selected history entry no longer exists.")
                job = jobs[index]
                snapshot = job.get("settings_snapshot") or job.get("settings") or {}
                kind = body.get("kind", "output")
                if kind == "logs":
                    values = [snapshot.get("logging_dir"), job.get("logging_dir")]
                else:
                    output_dir = snapshot.get("output_dir") or job.get("output_dir")
                    output_name = snapshot.get("output_name") or job.get("output_name") or job.get("name")
                    values = [Path(str(output_dir)) / str(output_name) if output_dir and output_name else output_dir]
                target = next((Path(str(value)).expanduser().resolve() for value in values if value and Path(str(value)).expanduser().exists()), None)
                if target is None:
                    raise FileNotFoundError(f"No existing {kind} path was recorded for this job.")
                if sys.platform == "win32":
                    os.startfile(target)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)])
                else:
                    subprocess.Popen(["xdg-open", str(target)])
                return self._json({"opened": str(target)})
            if self.path == "/api/jobs/import-found":
                return self._json(import_output_jobs(load_settings()))
            if self.path == "/api/jobs/clear":
                SUPERVISOR.clear_history()
                clear_desktop_history()
                return self._json({"cleared": True})
            if self.path == "/api/legacy/start":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
                process = subprocess.Popen(
                    [sys.executable, str(ROOT / "musubi_tuner_gui.py")],
                    cwd=ROOT,
                    creationflags=creationflags,
                )
                return self._json({"started": True, "pid": process.pid})
            self._error("Unknown API endpoint.", HTTPStatus.NOT_FOUND)
        except (DocumentConflictError, CaptionConflictError) as exc:
            self._error(exc, HTTPStatus.CONFLICT)
        except (MediaTokenError, PermissionError) as exc:
            self._error(exc, HTTPStatus.FORBIDDEN)
        except Exception as exc:
            self._error(exc)

    def _static(self, request_path):
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in candidate.parents and candidate != STATIC_ROOT.resolve():
            return self._error("Invalid path.", HTTPStatus.FORBIDDEN)
        if not candidate.is_file():
            candidate = STATIC_ROOT / "index.html"
        content = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, message, *args):
        print(f"[web-gui] {self.address_string()} - {message % args}")


def main():
    parser = argparse.ArgumentParser(description="Run the experimental Musubi web GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8675, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}"
    try:
        server = ThreadingHTTPServer((args.host, args.port), MusubiWebHandler)
    except OSError as exc:
        # A second launcher should be harmless when this GUI is already
        # running. Confirm the port belongs to our local server before asking
        # the default browser to reuse its existing window/tab.
        if exc.errno in {98, 10048}:
            try:
                with urllib.request.urlopen(f"{url}/api/health", timeout=0.6) as response:
                    healthy = response.status == 200
            except (OSError, urllib.error.URLError):
                healthy = False
            if healthy:
                print(f"Musubi modern GUI is already running at {url}")
                if not args.no_browser:
                    webbrowser.open(url, new=0)
                return
        raise
    print(f"Musubi modern GUI is available at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
