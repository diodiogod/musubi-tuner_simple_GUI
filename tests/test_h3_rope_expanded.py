import pytest
import torch

from musubi_tuner.minimax_h3 import model as compact_model
from musubi_tuner.minimax_h3_native import model as native_model


def _matmul_reference(hidden_states, rotation_table):
    pairs = rotation_table.shape[-3]
    rotary = torch.stack((hidden_states[..., :pairs], hidden_states[..., pairs : 2 * pairs]), dim=-1)
    rotary = torch.matmul(rotation_table, rotary.unsqueeze(-1)).squeeze(-1)
    rotary = torch.cat((rotary[..., 0], rotary[..., 1]), dim=-1)
    return torch.cat((rotary, hidden_states[..., 2 * pairs :]), dim=-1)


@pytest.mark.parametrize("implementation", [compact_model._apply_rope_split_half, native_model._apply_rope_split_half])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_expanded_rope_matches_matmul_forward_and_backward(implementation, dtype):
    torch.manual_seed(123)
    hidden = torch.randn(2, 7, 3, 12, dtype=dtype, requires_grad=True)
    angles = torch.rand(2, 7, 1, 4, dtype=torch.float32)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    table = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(2, 7, 1, 4, 2, 2).to(dtype)

    expected = _matmul_reference(hidden, table)
    actual = implementation(hidden, table)
    assert torch.equal(actual, expected)

    expected_grad = torch.autograd.grad(expected.float().square().sum(), hidden, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual.float().square().sum(), hidden)[0]
    assert torch.equal(actual_grad, expected_grad)
