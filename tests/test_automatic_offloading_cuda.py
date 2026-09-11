"""Opt-in GPU numerical gate; launch pytest via Accelerate on the intended GPU."""
import copy
import os
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from musubi_tuner.modules.automatic_offloading import AutomaticLoRAStreamOffloader

pytestmark = pytest.mark.skipif(os.getenv("H3_AUTO_SWAP_CUDA_TEST") != "1", reason="explicit CUDA test opt-in")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(32, 32, bias=False).requires_grad_(False)
        self.adapter = nn.Parameter(torch.randn(32, 32) * 0.001)

    def forward(self, x):
        return x + (self.base(x) + x @ self.adapter).tanh() * 0.1


@pytest.mark.parametrize("pinned", [False, True])
def test_transitions_backward_accumulation_and_frozen_masters(pinned):
    assert "4090" in torch.cuda.get_device_name(0), "Refusing to test on the secondary GPU"
    device = torch.device("cuda:0")
    torch.manual_seed(17)
    blocks = nn.ModuleList([Block() for _ in range(8)])
    reference = copy.deepcopy(blocks).to(device)
    offloader = AutomaticLoRAStreamOffloader("test", blocks, len(blocks), True, device,
                                           use_pinned_memory=pinned)
    offloader.prepare_block_devices_before_forward(blocks)
    masters = {i: t.clone() for i, t in offloader.cpu_flat.items()}
    adapters = [b.adapter for b in blocks]
    identities = [id(p) for p in adapters]
    optimizer = torch.optim.AdamW(adapters, lr=0.001)
    reference_optimizer = torch.optim.AdamW([b.adapter for b in reference], lr=0.001)
    for iteration, count in enumerate([8, 4, 2, 8, 6, 2, 7, 8]):
        offloader.reconfigure(count)
        offloader.begin_microbatch()
        with pytest.raises(RuntimeError, match="live microbatch"):
            offloader.reconfigure(4)
        # Primary plus auxiliary backward; leave gradients accumulated on odd steps.
        for auxiliary in range(2):
            x = torch.randn(2, 16 + iteration, 32, device=device, requires_grad=True)
            ref_x = x.detach().clone().requires_grad_(True)
            for i, block in enumerate(blocks):
                offloader.wait_for_block(i)
                x = checkpoint(block, x, use_reentrant=False)
                offloader.submit_move_blocks_forward(blocks, i)
            for block in reference:
                ref_x = checkpoint(block, ref_x, use_reentrant=False)
            loss = x.square().mean()
            ref_loss = ref_x.square().mean()
            torch.testing.assert_close(loss, ref_loss)
            loss.backward()
            ref_loss.backward()
        for block, ref in zip(blocks, reference):
            torch.testing.assert_close(block.adapter.grad, ref.adapter.grad, rtol=1e-5, atol=1e-7)
        if iteration % 2:
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        offloader.end_microbatch()
        if iteration == 3:
            offloader.set_forward_only(True)
            offloader.prepare_block_devices_before_forward(blocks)
            with torch.no_grad():
                preview = torch.randn(1, 4, 32, device=device)
                for i, block in enumerate(blocks):
                    offloader.wait_for_block(i)
                    preview = block(preview)
                    offloader.submit_move_blocks_forward(blocks, i)
            offloader.set_forward_only(False)
            offloader.prepare_block_devices_before_forward(blocks)
        assert identities == [id(b.adapter) for b in blocks]
        for i, original in masters.items():
            assert torch.equal(offloader.cpu_flat[i], original)
    offloader.copier.sync()
    for block, ref in zip(blocks, reference):
        torch.testing.assert_close(block.adapter, ref.adapter)


def test_failed_promotion_leaves_coherent_streaming_layout():
    assert "4090" in torch.cuda.get_device_name(0)
    device = torch.device("cuda:0")
    blocks = nn.ModuleList([Block() for _ in range(8)])
    offloader = AutomaticLoRAStreamOffloader("test", blocks, 8, True, device)
    offloader.prepare_block_devices_before_forward(blocks)
    original_to = torch.Tensor.to

    def fail_promotion(tensor, *args, **kwargs):
        if tensor.device.type == "cpu" and tensor.dtype == torch.uint8:
            raise torch.OutOfMemoryError("simulated graph-free promotion failure")
        return original_to(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "to", fail_promotion):
        with pytest.raises(torch.OutOfMemoryError, match="simulated"):
            offloader.reconfigure(4)
    assert offloader.S == 8
    assert not offloader.resident_flat
    offloader.reconfigure(4)
    assert offloader.S == 4
    offloader.copier.sync()
