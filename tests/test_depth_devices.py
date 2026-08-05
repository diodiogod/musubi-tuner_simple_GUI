import pytest
import torch

from musubi_tuner.perceptual.depth_devices import resolve_depth_vae_device


def test_secondary_depth_device_uses_other_visible_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert resolve_depth_vae_device("secondary", torch.device("cuda:0")) == torch.device("cuda:1")
    assert resolve_depth_vae_device("secondary", torch.device("cuda:1")) == torch.device("cuda:0")


def test_secondary_depth_device_requires_another_visible_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="only one CUDA device"):
        resolve_depth_vae_device("secondary", torch.device("cuda:0"))
