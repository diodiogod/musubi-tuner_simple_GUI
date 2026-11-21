import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

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

    assert rescale_lora(plan) == 1
    assert plan.output == tmp_path / "MOSS model_rescaled.safetensors"
    assert source.read_bytes() == source_bytes
    output = load_file(plan.output)
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
    with safe_open(plan.output, framework="pt") as reader:
        saved_metadata = reader.metadata()
    assert saved_metadata["trigger"] == metadata["trigger"]
    assert saved_metadata["ss_network_alpha"] == metadata["ss_network_alpha"]
    assert saved_metadata["sshs_model_hash"] != "original"
    assert saved_metadata["sshs_legacy_hash"] != "original"


def test_rescale_checks_strengths_and_preserves_existing_file_even_after_planning(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file({"x.lora_down.weight": torch.ones(2, 3), "x.lora_up.weight": torch.ones(4, 2)}, source)
    with pytest.raises(ValueError, match="Target strength"):
        plan_rescale(source, "", "1.8", "0")
    with pytest.raises(ValueError, match="Current strength"):
        plan_rescale(source, "", "nan")
    with pytest.raises(ValueError, match="different filename"):
        plan_rescale(source, source.name, "1.8")

    plan = plan_rescale(source, "publish.safetensors", "1.8")
    plan.output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        rescale_lora(plan)
    assert plan.output.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".lora-rescale-*"))


@pytest.mark.parametrize("extra_key", ["x.dora_scale", "x.diff", "y.lora_up.weight"])
def test_rescale_rejects_partial_or_nonlinear_adapters(tmp_path, extra_key):
    source = tmp_path / "source.safetensors"
    save_file({"x.lora_down.weight": torch.ones(2, 3), "x.lora_up.weight": torch.ones(4, 2), extra_key: torch.ones(4)}, source)
    plan = plan_rescale(source, "", "1.8")
    with pytest.raises(ValueError, match="unsupported tensor"):
        rescale_lora(plan)
    assert not plan.output.exists()
