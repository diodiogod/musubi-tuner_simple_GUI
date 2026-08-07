from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import bootstrap_environment as bootstrap


def test_choose_torch_channel_prefers_cu128_for_current_drivers(monkeypatch):
    monkeypatch.delenv("MUSUBI_CUDA", raising=False)
    assert bootstrap.choose_torch_channel((12, 8)) == "cu128"
    assert bootstrap.choose_torch_channel((13, 1)) == "cu128"


def test_choose_torch_channel_uses_cu124_for_older_driver(monkeypatch):
    monkeypatch.delenv("MUSUBI_CUDA", raising=False)
    assert bootstrap.choose_torch_channel((12, 4)) == "cu124"
    assert bootstrap.choose_torch_channel(None) == "cu124"


def test_choose_torch_channel_honors_override(monkeypatch):
    monkeypatch.setenv("MUSUBI_CUDA", "cu130")
    assert bootstrap.choose_torch_channel((12, 4)) == "cu130"


def test_choose_torch_channel_rejects_unknown_override(monkeypatch):
    monkeypatch.setenv("MUSUBI_CUDA", "latest")
    with pytest.raises(ValueError, match="Unsupported MUSUBI_CUDA"):
        bootstrap.choose_torch_channel((13, 0))


def test_detected_driver_cuda_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="Driver Version: 580.0  CUDA Version: 13.0"),
    )
    assert bootstrap.detected_driver_cuda() == (True, (13, 0))


def test_fast_environment_check_uses_module_discovery(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    assert bootstrap.environment_available(tmp_path / "python") is True
    assert "find_spec" in captured["command"][2]
    assert "import torch" not in captured["command"][2]
