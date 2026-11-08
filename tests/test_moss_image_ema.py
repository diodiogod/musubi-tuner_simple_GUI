from argparse import Namespace
from copy import deepcopy

import pytest
import torch

from modern_gui.prompt_preview import resolve_preview_lora
from musubi_tuner.moss_image.ema import AdapterEMA, validate_ema_resume
from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer


def test_ema_averages_only_trainable_adapter_and_restores_after_failure():
    model = torch.nn.Linear(2, 2)
    model.weight.data.zero_()
    model.bias.requires_grad_(False)
    ema = AdapterEMA(model, decay=0.98, device="cpu")
    assert set(ema.shadow) == {"weight"}
    model.weight.data.fill_(2)
    ema.update()
    assert ema.num_updates == 1
    torch.testing.assert_close(ema.shadow["weight"], torch.full_like(model.weight, 0.04))
    with pytest.raises(RuntimeError, match="sample failed"):
        with ema.average_parameters():
            torch.testing.assert_close(model.weight, ema.shadow["weight"])
            with ema.average_parameters():
                torch.testing.assert_close(model.weight, ema.shadow["weight"])
            raise RuntimeError("sample failed")
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2))


def test_ema_supports_full_matrix_lokr_without_averaging_the_base():
    from musubi_tuner.networks.lokr import LoKrModule
    base = torch.nn.Linear(8, 8).requires_grad_(False)
    base_weights = base.weight.detach().clone()
    adapter = LoKrModule("lora_unet_test", base, lora_dim=8, alpha=8, factor=2)
    adapter.apply_to()
    assert adapter.use_w2
    ema = AdapterEMA(adapter, decay=0.5)
    adapter.lokr_w2.data.fill_(2)
    ema.update()
    with ema.average_parameters():
        torch.testing.assert_close(adapter.lokr_w2, torch.ones_like(adapter.lokr_w2))
        assert torch.isfinite(base(torch.ones(1, 8))).all()
    torch.testing.assert_close(adapter.lokr_w2, torch.full_like(adapter.lokr_w2, 2))
    assert torch.equal(base.weight, base_weights)


def test_ema_resume_matches_uninterrupted_averaging_and_validates_before_copy():
    model = torch.nn.Linear(2, 2)
    ema = AdapterEMA(model)
    model.weight.data.add_(1)
    ema.update()
    resumed = AdapterEMA(model)
    resumed.load_state_dict(deepcopy(ema.state_dict()))
    model.weight.data.add_(2)
    ema.update()
    resumed.update()
    for name in ema.shadow:
        torch.testing.assert_close(resumed.shadow[name], ema.shadow[name])
    invalid = deepcopy(ema.state_dict())
    invalid["shadow"]["weight"].fill_(float("nan"))
    before = deepcopy(resumed.state_dict())
    with pytest.raises(ValueError, match="Invalid saved"):
        resumed.load_state_dict(invalid)
    for name in before["shadow"]:
        assert torch.equal(resumed.shadow[name], before["shadow"][name])


def test_ema_updates_once_per_successful_optimizer_step():
    trainer = MossImageNetworkTrainer()
    trainer._ema = AdapterEMA(torch.nn.Linear(2, 2))
    args = Namespace(weight_noise_sigma=0)
    trainer.on_post_optimizer_step(args, Namespace(optimizer_step_was_skipped=False), None, None, False, 0)
    trainer.on_post_optimizer_step(args, Namespace(optimizer_step_was_skipped=True), None, None, True, 0)
    assert trainer._ema.num_updates == 0
    trainer.on_post_optimizer_step(args, Namespace(optimizer_step_was_skipped=False), None, None, True, 0)
    assert trainer._ema.num_updates == 1


def test_ema_full_resume_settings_and_preview_companion(tmp_path):
    folder = tmp_path / "state"
    folder.mkdir()
    validate_ema_resume(folder, False, 0.98)
    with pytest.raises(ValueError, match="LoRA weights"):
        validate_ema_resume(folder, True, 0.98)
    ema = AdapterEMA(torch.nn.Linear(2, 2))
    ema.save_manifest(None, None, folder)
    validate_ema_resume(folder, True, 0.98)
    with pytest.raises(ValueError, match="configuration"):
        validate_ema_resume(folder, True, 0.99)
    (folder / "model.safetensors").write_bytes(b"live")
    settings = {"preview_use_lora": True, "starting_point_mode": "state", "resume_path": str(folder)}
    with pytest.raises(ValueError, match="ema.safetensors"):
        resolve_preview_lora(settings)
    averaged = folder / "ema.safetensors"
    averaged.write_bytes(b"averaged")
    assert resolve_preview_lora(settings) == str(averaged)


def test_gui_ema_and_optimizer_controls_are_scoped_and_explained():
    from backends.moss_settings import DEFAULTS, MODE
    from modern_gui.settings import settings_schema
    schema = settings_schema({**DEFAULTS, "training_mode": MODE, "optimizer_type": "Automagic3"})
    fields = {field["key"]: (section["id"], field) for section in schema["sections"] for field in section["fields"]}
    assert fields["moss_ema_enabled"][0] == "regularization"
    assert fields["moss_automagic3_min_lr"][0] == "optimization"
    assert fields["moss_ema_enabled"][1]["modes"] == [MODE]
    assert fields["moss_ema_decay"][1]["step"] == 0.001
    assert "same smoothing" in fields["moss_ema_enabled"][1]["help_detail"]
    assert "fresh optimizer" in fields["moss_ema_enabled"][1]["help_detail"]
    assert "fused=False" in fields["optimizer_type"][1]["help_detail"]
