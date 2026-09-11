from argparse import Namespace

import pytest
import torch

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader


def test_old_arguments_preserve_fixed_defaults():
    config = BlockSwapConfig.from_args(Namespace(), torch.device("cuda:0"), True)
    assert not config.automatic
    assert not config.h2d_only


def test_automatic_enables_frozen_streaming_and_requires_checkpointing():
    args = Namespace(auto_block_swap=True, gradient_checkpointing=False)
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        BlockSwapConfig.from_args(args, torch.device("cuda:0"), True)
    args.gradient_checkpointing = True
    config = BlockSwapConfig.from_args(args, torch.device("cuda:0"), True)
    assert config.automatic and config.h2d_only


def test_factory_rejects_other_architectures_before_allocating():
    config = BlockSwapConfig(torch.device("cuda:0"), True, automatic=True)
    with pytest.raises(ValueError, match="only for MiniMax H3"):
        create_offloader("wan", [], 0, 0, config)


def test_automatic_does_not_silently_ignore_custom_ring_size():
    with pytest.raises(ValueError, match="ring_size 2"):
        BlockSwapConfig.from_args(Namespace(auto_block_swap=True, gradient_checkpointing=True,
                                             block_swap_ring_size=1), torch.device("cuda:0"), True)
