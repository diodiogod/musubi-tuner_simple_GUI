"""Tests for gradient metrics collection (grad/norm, grad/mean_norm, grad/max)."""

import pytest
import torch
import torch.nn as nn

from musubi_tuner.training.trainer_base import NetworkTrainer


@pytest.fixture
def trainer():
    return NetworkTrainer()


def _params_with_grads(values):
    params = []
    for vals in values:
        parameter = nn.Parameter(torch.zeros(len(vals)))
        parameter.grad = torch.tensor(vals)
        params.append(parameter)
    return params


def test_grad_norm_single_param(trainer):
    metrics = trainer.collect_grad_metrics(_params_with_grads([[3.0, 4.0]]))
    assert metrics["grad/norm"] == pytest.approx(5.0)


def test_grad_norm_multiple_params(trainer):
    metrics = trainer.collect_grad_metrics(_params_with_grads([[1.0, 0.0], [0.0, 1.0]]))
    assert metrics["grad/norm"] == pytest.approx(2.0**0.5, rel=1e-5)


def test_grad_max_across_params(trainer):
    metrics = trainer.collect_grad_metrics(_params_with_grads([[1.0, 2.0], [3.0, -9.0]]))
    assert metrics["grad/max"] == pytest.approx(9.0)


def test_empty_metrics_when_no_grads(trainer):
    assert trainer.collect_grad_metrics([nn.Parameter(torch.zeros(4))]) == {}


def test_grad_mean_norm_multiple_params(trainer):
    metrics = trainer.collect_grad_metrics(_params_with_grads([[3.0, 4.0], [4.0, 3.0]]))
    assert metrics["grad/mean_norm"] == pytest.approx(5.0)


def test_skips_params_without_grad(trainer):
    with_grad = _params_with_grads([[3.0, 4.0]])[0]
    metrics = trainer.collect_grad_metrics([with_grad, nn.Parameter(torch.zeros(2))])
    assert metrics["grad/norm"] == pytest.approx(5.0)
    assert metrics["grad/max"] == pytest.approx(4.0)


def test_log_grad_metrics_flag_default_off():
    from musubi_tuner.training.parser_common import setup_parser_common

    parser = setup_parser_common()
    args, _ = parser.parse_known_args([])
    assert args.log_grad_metrics is False
    args, _ = parser.parse_known_args(["--log_grad_metrics"])
    assert args.log_grad_metrics is True
