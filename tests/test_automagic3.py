from argparse import Namespace
from copy import deepcopy

import pytest
import torch

from backends.moss_image import build_commands, training_options
from musubi_tuner.optimizers.automagic3 import Automagic3, Automagic3Scheduler
from musubi_tuner.training.trainer_base import NetworkTrainer


def run_signs(signs, **kwargs):
    p = torch.nn.Parameter(torch.ones(3, 5))
    opt = Automagic3([p], lr=1e-4, polarity_history=4, **kwargs)
    scheduler = Automagic3Scheduler(opt)
    for sign in signs:
        p.grad = torch.full_like(p, float(sign))
        opt.step()
        scheduler.step()
        opt.zero_grad()
    return p, opt, scheduler


def test_adapted_rate_responds_to_direction_and_scheduler_reports_it():
    _, consistent, schedule = run_signs([1] * 16, min_lr=1e-6, max_lr=5e-4)
    assert 1e-4 < consistent.get_avg_learning_rate() <= 5.000001e-4
    assert schedule.get_last_lr() == consistent.get_learning_rates()
    _, alternating, _ = run_signs([1, -1] * 8, min_lr=5e-5, max_lr=5e-4)
    assert 4.999999e-5 <= alternating.get_avg_learning_rate() < 1e-4


def test_backward_accumulation_and_clipping_happen_before_the_update():
    p = torch.nn.Parameter(torch.ones(3))
    opt = Automagic3([p], lr=1e-4)
    before = p.detach().clone()
    p.sum().backward()
    (p * 2).sum().backward()
    assert torch.equal(p, before)
    torch.testing.assert_close(p.grad, torch.full_like(p, 3))
    torch.nn.utils.clip_grad_norm_([p], 0.1)
    clipped = p.grad.clone()
    opt.step()
    torch.testing.assert_close(opt.state[p]["exp_avg_sq"], clipped.square() * (1 - opt.defaults["beta2"]))
    assert not torch.equal(p, before)


def test_resume_preserves_adaptive_state_and_next_update():
    p, opt, schedule = run_signs([1, 1, -1, 1] * 3, min_lr=1e-6, max_lr=1e-3)
    resumed_p = torch.nn.Parameter(p.detach().clone())
    resumed = Automagic3([resumed_p], lr=1e-4, polarity_history=4, min_lr=1e-6, max_lr=1e-3)
    resumed.load_state_dict(deepcopy(opt.state_dict()))
    resumed_schedule = Automagic3Scheduler(resumed)
    resumed_schedule.load_state_dict(deepcopy(schedule.state_dict()))
    assert resumed.state[resumed_p]["sign_history"].dtype == torch.uint8
    for sign in [1, -1, -1, -1]:
        for param, optimizer, scheduler in [(p, opt, schedule), (resumed_p, resumed, resumed_schedule)]:
            param.grad = torch.full_like(param, float(sign))
            optimizer.step()
            scheduler.step()
        torch.testing.assert_close(p, resumed_p)
        assert resumed.get_avg_learning_rate() == pytest.approx(opt.get_avg_learning_rate(), rel=1e-6)


def test_musubi_rejects_backward_updates_and_low_precision_adapters():
    with pytest.raises(ValueError, match="fused=False"):
        Automagic3([torch.nn.Parameter(torch.ones(2))], fused=True)
    with pytest.raises(ValueError, match="FP32"):
        Automagic3([torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))])


def test_resuming_adam_state_as_automagic_fails_before_mutating_optimizer():
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = Automagic3([p])
    with pytest.raises(ValueError, match="LoRA weights"):
        optimizer.load_state_dict(torch.optim.AdamW([p]).state_dict())
    assert not optimizer.state


def test_gui_rejects_switching_optimizer_on_new_full_state(tmp_path):
    import json
    (tmp_path / "moss_optimizer.json").write_text(json.dumps({"optimizer_type": "adamw8bit"}), encoding="utf-8")
    with pytest.raises(ValueError, match="same optimizer"):
        training_options({"resume_path": str(tmp_path), "optimizer_type": "Automagic3"})


def test_adaptive_optimizer_rejects_external_scheduler():
    optimizer = Automagic3([torch.nn.Parameter(torch.ones(2))])
    with pytest.raises(ValueError, match="constant"):
        NetworkTrainer().get_lr_scheduler(Namespace(lr_scheduler="cosine", lr_warmup_steps=0), optimizer, 1)


def test_moss_command_sends_all_optimizer_options_in_one_argument_group():
    from musubi_tuner.moss_image_train_network import moss_setup_parser, setup_parser_common
    settings = {"optimizer_type": "Automagic3", "lr_scheduler": "constant", "learning_rate": "0.0005",
                "optimizer_args": "eps=1e-25; beta2=0.98", "output_name": "test", "output_dir": "."}
    original = dict(settings)
    cmd = build_commands(settings)[0]
    assert cmd.count("--optimizer_args") == 1
    args = moss_setup_parser(setup_parser_common()).parse_args(cmd[cmd.index("src/musubi_tuner/moss_image_train_network.py") + 1:])
    options = dict(token.split("=", 1) for token in args.optimizer_args)
    assert options == {"eps": "1e-25", "beta2": "0.98", "min_lr": "5e-05", "max_lr": "0.001", "weight_decay": "0.00001"}
    assert settings == original


@pytest.mark.parametrize("extra", [{"lr_scheduler": "cosine"}, {"lr_warmup_steps": "5"},
                                  {"optimizer_args": "min_lr=0.1"}, {"optimizer_args": "fused=True"},
                                  {"optimizer_args": "betas=(0.9,0.99)"}, {"optimizer_args": "eps=-1"},
                                  {"moss_automagic3_min_lr": "0.002"}, {"full_bf16": True}])
def test_moss_validation_rejects_conflicting_automagic_options(extra):
    with pytest.raises(ValueError):
        training_options({"optimizer_type": "Automagic3", "lr_scheduler": "constant", "learning_rate": "0.0005", **extra})


def test_multiple_existing_adam_options_reach_cli_without_being_dropped():
    from backends._common import build_common_train_args
    from musubi_tuner.training.parser_common import setup_parser_common
    cmd = []
    build_common_train_args(cmd, {"optimizer_args": "betas=(0.9,0.99), weight_decay=0.01 eps=1e-8"})
    args = setup_parser_common().parse_args(cmd)
    assert args.optimizer_args == ["betas=(0.9,0.99)", "weight_decay=0.01", "eps=1e-8"]
