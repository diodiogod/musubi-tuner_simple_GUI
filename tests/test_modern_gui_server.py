import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from modern_gui import server as server_module
from modern_gui.server import MusubiWebHandler


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    return json.load(urllib.request.urlopen(request))


def test_server_hosts_app_and_core_read_only_apis():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        health = request_json(root + "/api/health")
        settings = request_json(root + "/api/settings")
        parsed = request_json(
            root + "/api/dataset/parse",
            {"text": '[[datasets]]\nimage_directory = "images"\nresolution = [512, 512]\n'},
        )
        html = urllib.request.urlopen(root + "/").read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()

    assert health["ok"] is True
    assert "schema" in settings
    assert parsed["datasets"][0]["resolution"] == [512, 512]
    assert "Musubi Studio" in html


def test_server_native_folder_drop_and_dataset_add(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    monkeypatch.setattr("modern_gui.path_drop.choose_directories", lambda title: [str(image_dir)])
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    source = """[general]
resolution = [768, 768]
caption_extension = ".txt"

[[datasets]]
image_directory = "existing"
"""
    try:
        dropped = request_json(root + "/api/path/drop", {"title": "Drop images"})
        added = request_json(
            root + "/api/dataset/add",
            {"text": source, "path": "dataset.toml", "kind": "image", "source_path": dropped["path"]},
        )
    finally:
        server.shutdown()
        server.server_close()

    assert dropped["path"] == str(image_dir)
    assert dropped["paths"] == [str(image_dir)]
    assert added["datasets"][-1]["source"] == str(image_dir)
    assert added["datasets"][-1]["resolution"] == [768, 768]
    assert added["datasets"][-1]["value_origins"]["resolution"] == "general"


def test_server_expands_dataset_subfolders(tmp_path):
    root = tmp_path / "car"
    (root / "1_low_quality").mkdir(parents=True)
    (root / "4_high_quality").mkdir()
    Image.new("RGB", (32, 24), "red").save(root / "1_low_quality" / "low.png")
    Image.new("RGB", (32, 24), "blue").save(root / "4_high_quality" / "high.png")
    source = f'''[[datasets]]
image_directory = {str(root)!r}
resolution = 512
'''
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_port}"
    try:
        split = request_json(root_url + "/api/dataset/split-subfolders", {"text": source, "index": 0})
    finally:
        server.shutdown()
        server.server_close()

    assert len(split["datasets"]) == 2
    assert split["subfolder_scan"]["removed_parent"] is True


def test_server_toggles_dataset_without_deleting_its_settings():
    source = """[general]
caption_extension = ".txt"

[[datasets]]
image_directory = "images"
resolution = [512, 512]
future_option = "preserve-me"
"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        disabled = request_json(
            root + "/api/dataset/toggle-disabled",
            {"text": source, "index": 0, "disabled": True},
        )
        restored = request_json(
            root + "/api/dataset/toggle-disabled",
            {"text": disabled["text"], "index": 0, "disabled": False},
        )
    finally:
        server.shutdown()
        server.server_close()

    assert disabled["datasets"] == []
    assert len(disabled["disabled_datasets"]) == 1
    assert restored["datasets"][0]["source"] == "images"
    assert restored["datasets"][0]["raw_values"]["future_option"] == "preserve-me"


def test_server_lists_serves_and_updates_dataset_media(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "item.png"
    Image.new("RGB", (32, 24), "blue").save(image_path)
    source = f"""[general]
caption_extension = ".txt"

[[datasets]]
image_directory = {str(image_dir)!r}
resolution = 512
"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        media = request_json(root + "/api/dataset/media", {"text": source, "index": 0})
        item = media["items"][0]
        preview = urllib.request.urlopen(
            root + "/api/dataset/media-file?token=" + item["token"]
        ).read()
        saved = request_json(
            root + "/api/dataset/caption",
            {
                "token": item["token"],
                "caption": "caption through the API",
                "expected_revision": item["caption_revision"],
            },
        )
        hostile = urllib.request.Request(
            root + "/api/dataset/caption",
            data=json.dumps(
                {
                    "token": item["token"],
                    "caption": "hostile overwrite",
                    "expected_revision": saved["caption_revision"],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": "https://example.invalid"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(hostile)
        denied_status = denied.value.code
    finally:
        server.shutdown()
        server.server_close()

    assert preview == image_path.read_bytes()
    assert saved["caption_state"] == "present"
    assert (image_dir / "item.txt").read_text(encoding="utf-8") == "caption through the API"
    assert denied_status == 403


def test_server_supports_video_byte_ranges(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "clip.mp4"
    video_path.write_bytes(b"0123456789")
    (video_dir / "clip.txt").write_text("caption", encoding="utf-8")
    source = f"""[general]
caption_extension = ".txt"

[[datasets]]
video_directory = {str(video_dir)!r}
resolution = [512, 512]
target_frames = [1]
"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        media = request_json(root + "/api/dataset/media", {"text": source, "index": 0})
        request = urllib.request.Request(
            root + "/api/dataset/media-file?token=" + media["items"][0]["token"],
            headers={"Range": "bytes=2-5"},
        )
        response = urllib.request.urlopen(request)
        content = response.read()
        status = response.status
        content_range = response.headers["Content-Range"]
    finally:
        server.shutdown()
        server.server_close()

    assert status == 206
    assert content == b"2345"
    assert content_range == "bytes 2-5/10"


@pytest.mark.parametrize(
    "path",
    [
        "/api/settings",
        "/api/dataset/toggle-disabled",
        "/api/prompt-library/delete",
        "/api/prompts/preview",
        "/api/jobs/start",
    ],
)
def test_server_rejects_cross_origin_state_changes_before_parsing(path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    request = urllib.request.Request(
        root + path,
        data=b"this is deliberately not JSON",
        headers={"Content-Type": "text/plain", "Origin": "https://example.invalid"},
        method="POST",
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(request)
        status = denied.value.code
    finally:
        server.shutdown()
        server.server_close()

    assert status == 403


def test_server_rejects_a_different_loopback_origin():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    request = urllib.request.Request(
        root + "/api/settings",
        data=b"this is deliberately not JSON",
        headers={
            "Content-Type": "text/plain",
            "Origin": f"http://127.0.0.1:{server.server_port + 1}",
        },
        method="POST",
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(request)
        status = denied.value.code
    finally:
        server.shutdown()
        server.server_close()

    assert status == 403


def _preview_settings(tmp_path):
    models = {}
    for name in ("raw.safetensors", "turbo.safetensors", "vae.safetensors", "text.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"x")
        models[name] = str(path)
    output_root = tmp_path / "outputs"
    run_root = output_root / "portrait"
    run_root.mkdir(parents=True)
    resolved_lora = run_root / "portrait.safetensors"
    resolved_lora.write_bytes(b"lora")
    return {
        "training_mode": "Krea 2",
        "krea2_dit_model": models["raw.safetensors"],
        "krea2_turbo_dit": models["turbo.safetensors"],
        "vae_model": models["vae.safetensors"],
        "krea2_text_encoder": models["text.safetensors"],
        "output_dir": str(output_root),
        "output_name": "portrait",
        "attention_mechanism": "sdpa",
    }, resolved_lora


def test_prompt_preview_reports_exact_turbo_and_resolved_lora_metadata(monkeypatch, tmp_path):
    settings, resolved_lora = _preview_settings(tmp_path)
    captured = {}

    monkeypatch.setattr(
        server_module.SUPERVISOR,
        "snapshot",
        lambda after=0: {"active": None, "log": [], "last_log_id": 0},
    )

    def fake_start(commands, *, name, mode, kind, settings, completion_context=None):
        captured.update(
            {
                "commands": commands,
                "name": name,
                "mode": mode,
                "kind": kind,
                "settings": settings,
                "completion_context": completion_context,
            }
        )
        return {"id": "preview-job", "status": "starting", "mode": mode}

    monkeypatch.setattr(server_module.SUPERVISOR, "start_commands", fake_start)
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        payload = request_json(
            root + "/api/prompts/preview",
            {
                "settings": settings,
                "prompts": [
                    {"prompt": "one", "enabled": True},
                    {"prompt": "two", "enabled": True},
                ],
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    prompt_command = captured["commands"][0]
    prompt_file = prompt_command[prompt_command.index("--from_file") + 1]
    assert payload["preview_mode"] == "Krea 2 Turbo"
    assert payload["network_weights"] == str(resolved_lora)
    assert captured["mode"] == "Krea 2 Turbo"
    assert captured["completion_context"]["mode"] == "Krea 2 Turbo"
    assert captured["completion_context"]["network_weights"] == str(resolved_lora)
    assert Path(prompt_file).read_text(encoding="utf-8") == "one\ntwo"


def test_prompt_preview_cleans_unique_batch_file_if_start_loses_race(monkeypatch, tmp_path):
    settings, _resolved_lora = _preview_settings(tmp_path)
    monkeypatch.setattr(
        server_module.SUPERVISOR,
        "snapshot",
        lambda after=0: {"active": None, "log": [], "last_log_id": 0},
    )
    monkeypatch.setattr(
        server_module.SUPERVISOR,
        "start_commands",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("A web GUI job is already active.")),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as failed:
            request_json(
                root + "/api/prompts/preview",
                {
                    "settings": settings,
                    "prompts": [
                        {"prompt": "one", "enabled": True},
                        {"prompt": "two", "enabled": True},
                    ],
                },
            )
        status = failed.value.code
    finally:
        server.shutdown()
        server.server_close()

    save_path = Path(settings["output_dir"]) / settings["output_name"] / "sample_test"
    assert status == 400
    assert not list(save_path.glob("preview_prompts_*.txt"))
    assert not list(save_path.glob(".preview-prompts-*.tmp"))


def test_prompt_preview_checks_active_job_before_creating_files(monkeypatch, tmp_path):
    output_root = tmp_path / "must-not-be-created"
    monkeypatch.setattr(
        server_module.SUPERVISOR,
        "snapshot",
        lambda after=0: {
            "active": {"id": "active-job", "status": "running"},
            "log": [],
            "last_log_id": 0,
        },
    )
    monkeypatch.setattr(
        server_module.SUPERVISOR,
        "start_commands",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preview launch was reached")),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), MusubiWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as failed:
            request_json(
                root + "/api/prompts/preview",
                {
                    "settings": {
                        "training_mode": "Krea 2",
                        "output_dir": str(output_root),
                        "output_name": "preview",
                    },
                    "prompts": [
                        {"prompt": "one", "enabled": True},
                        {"prompt": "two", "enabled": True},
                    ],
                },
            )
        status = failed.value.code
        error = json.load(failed.value)["error"]
    finally:
        server.shutdown()
        server.server_close()

    assert status == 400
    assert error == "A web GUI job is already active."
    assert not output_root.exists()
