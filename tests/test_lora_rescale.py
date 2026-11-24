import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from backends import lora_rescale
from backends.lora_rescale import plan_rescale, rescale_lora


@pytest.mark.parametrize(
    "down_name, up_name, alpha, current, target",
    [
        ("lora_down.weight", "lora_up.weight", 2, "1,8", "1.0"),
        ("lora_A.weight", "lora_B.weight", None, "1.8", "1,0"),
        ("lora_A.default.weight", "lora_B.default.weight", None, "-1.8", "0.5"),
    ],
)
def test_rescale_preserves_weights_and_comfy_patch(tmp_path, down_name, up_name, alpha, current, target):
    prefix = "lora_unet_layers_0_proj_q"
    down_key, up_key, alpha_key = (f"{prefix}.{name}" for name in (down_name, up_name, "alpha"))
    generator = torch.Generator().manual_seed(231)
    tensors = {
        down_key: torch.randn(4, 16, generator=generator).bfloat16(),
        up_key: torch.randn(12, 4, generator=generator).bfloat16(),
    }
    if alpha is not None:
        tensors[alpha_key] = torch.tensor(alpha, dtype=torch.bfloat16)
    source = tmp_path / "MOSS model.safetensors"
    metadata = {"sshs_model_hash": "original", "sshs_legacy_hash": "original", "ss_network_alpha": "2", "trigger": "concept"}
    save_file(tensors, source, metadata=metadata)
    source_bytes = source.read_bytes()
    plan = plan_rescale(source, "", current, target)

    output_path, layers = rescale_lora(plan)
    assert layers == 1
    assert output_path == tmp_path / "MOSS model_rescaled.safetensors"
    assert source.read_bytes() == source_bytes
    output = load_file(output_path)
    assert set(output) == {down_key, up_key, alpha_key}
    for key in (down_key, up_key):
        assert output[key].dtype == tensors[key].dtype
        assert torch.equal(output[key], tensors[key])

    # ComfyUI casts the stored matrices to its compute dtype and applies
    # strength * (alpha / rank) to the resulting delta.
    rank = tensors[down_key].shape[0]
    original_delta = plan.current * (alpha / rank if alpha is not None else 1.0) * (
        tensors[up_key].float() @ tensors[down_key].float()
    )
    rescaled_delta = plan.target * (output[alpha_key].item() / rank) * (
        output[up_key].float() @ output[down_key].float()
    )
    assert torch.equal(rescaled_delta, original_delta)
    with safe_open(output_path, framework="pt") as reader:
        saved_metadata = reader.metadata()
    assert saved_metadata["trigger"] == metadata["trigger"]
    assert saved_metadata["ss_network_alpha"] == metadata["ss_network_alpha"]
    assert saved_metadata["sshs_model_hash"] != "original"
    assert saved_metadata["sshs_legacy_hash"] != "original"


@pytest.fixture
def source_lora(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file({"x.lora_down.weight": torch.ones(2, 3), "x.lora_up.weight": torch.ones(4, 2)}, source)
    return source


def test_rescale_checks_strengths(source_lora):
    with pytest.raises(ValueError, match="Target strength"):
        plan_rescale(source_lora, "", "1.8", "0")
    with pytest.raises(ValueError, match="Current strength"):
        plan_rescale(source_lora, "", "nan")


@pytest.mark.parametrize("requested, stem", [("", "source_rescaled"), ("publish.safetensors", "publish"), ("source.safetensors", "source")])
def test_rescale_numbers_taken_names(source_lora, tmp_path, requested, stem):
    taken = tmp_path / f"{stem}.safetensors"
    if taken != source_lora:
        taken.write_bytes(b"existing")
    original = taken.read_bytes()
    (tmp_path / f"{stem}_2.safetensors").mkdir()
    plan = plan_rescale(source_lora, requested, "1.8")

    output, count = rescale_lora(plan)

    assert (output, count) == (tmp_path / f"{stem}_3.safetensors", 1)
    assert output.is_file()
    assert taken.read_bytes() == original
    assert (tmp_path / f"{stem}_2.safetensors").is_dir()


def test_rescale_retries_if_another_export_claims_the_name(source_lora, tmp_path, monkeypatch):
    plan = plan_rescale(source_lora, "publish.safetensors", "1.8")
    publisher, method = (Path, "rename") if os.name == "nt" else (os, "link")
    publish = getattr(publisher, method)

    def race(staging, destination):
        if destination == plan.output:
            destination.write_bytes(b"competing export")
        return publish(staging, destination)

    monkeypatch.setattr(publisher, method, race)
    actual, _ = rescale_lora(plan)
    assert actual == tmp_path / "publish_2.safetensors"
    assert plan.output.read_bytes() == b"competing export"
    assert not list(tmp_path.glob(".lora-rescale-*"))


def test_gui_export_remembers_source_and_both_strengths(source_lora, tmp_path, monkeypatch, capsys):
    preferences_path = tmp_path / "preferences.json"
    monkeypatch.setattr(lora_rescale, "PREFERENCES_PATH", preferences_path)
    assert lora_rescale.load_rescale_preferences() == {"input": "", "current_strength": "", "target_strength": "1.0"}

    plan = plan_rescale(source_lora, "", "1,8", "0,5")
    plan.output.touch()
    monkeypatch.setattr(sys, "argv", ["rescale_lora", *plan.command()[3:]])
    lora_rescale.main()

    assert lora_rescale.load_rescale_preferences() == {
        "input": str(source_lora), "current_strength": "1.8", "target_strength": "0.5",
    }
    saved_line = capsys.readouterr().out.splitlines()[-1]
    assert lora_rescale.rescale_saved_path(saved_line) == tmp_path / "source_rescaled_2.safetensors"

    preferences_path.write_text("{truncated", encoding="utf-8")
    assert lora_rescale.load_rescale_preferences() == {"input": "", "current_strength": "", "target_strength": "1.0"}


@pytest.mark.parametrize("extra_key", ["x.dora_scale", "x.diff", "y.lora_up.weight"])
def test_rescale_rejects_partial_or_nonlinear_adapters(tmp_path, extra_key):
    source = tmp_path / "source.safetensors"
    save_file({"x.lora_down.weight": torch.ones(2, 3), "x.lora_up.weight": torch.ones(4, 2), extra_key: torch.ones(4)}, source)
    plan = plan_rescale(source, "", "1.8")
    with pytest.raises(ValueError, match="unsupported tensor"):
        rescale_lora(plan)
    assert not plan.output.exists()
