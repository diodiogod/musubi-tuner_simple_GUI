# Reference parity tests adapted from kohya-ss/sd-scripts (Apache-2.0).
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from musubi_tuner.moss_image import models as moss_image_models


# region reference implementation (per-sample loop, copied from the official pipeline_moss_image.py, Apache-2.0)


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        exponent = torch.arange(self.num_bands, dtype=torch.float32) / self.num_bands
        self.register_buffer("bands", 1 / (self.base**exponent))

    def forward(self, height, width, text_length, device):
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class ReferenceMossImage(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, num_layers, text_dim, freq_dim, latent_channels, rope_base):
        super().__init__()
        self.freq_dim = freq_dim
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def forward(self, hidden_states, timestep, encoder_hidden_states, mask, skip_layers=()):
        skip_layers = frozenset(skip_layers)
        batch, _, height, width = hidden_states.shape
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1).expand(batch)
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding).float()
        mod = self.shared_block_modulation(time).view(batch, 6, -1)
        head = self.video_head_modulation(time).view(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index : index + 1])
            x = self.final_norm_video(x[:, : height * width])
            x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            x = self.final_linear_video(x)
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)


# endregion


SMALL_CONFIG = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)


def _make_models(split_attn: bool):
    torch.manual_seed(0)
    reference = ReferenceMossImage(**SMALL_CONFIG).eval()
    model = moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="torch", split_attn=split_attn).eval()
    missing, unexpected = model.load_state_dict(reference.state_dict(), strict=True), None
    return reference, model


def _make_inputs():
    torch.manual_seed(1)
    batch, height, width = 2, 3, 5
    lens = [4, 7]
    x = torch.randn(batch, SMALL_CONFIG["latent_channels"], height, width)
    context = torch.randn(batch, max(lens), SMALL_CONFIG["text_dim"])
    mask = torch.zeros(batch, max(lens), dtype=torch.int32)
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    context = context * mask[..., None]  # padded positions are zero like the official pipeline
    timesteps = torch.tensor([750.0, 120.0])
    return x, timesteps, context, mask


@pytest.mark.parametrize("split_attn", [False, True])
def test_batched_forward_matches_reference_loop(split_attn):
    reference, model = _make_models(split_attn)
    x, timesteps, context, mask = _make_inputs()
    with torch.no_grad():
        expected = reference(x, timesteps, context, mask)
        actual = model(x, timesteps, context, mask)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_batched_forward_matches_reference_with_skip_layers():
    reference, model = _make_models(False)
    x, timesteps, context, mask = _make_inputs()
    with torch.no_grad():
        expected = reference(x, timesteps, context, mask, skip_layers=(1,))
        actual = model(x, timesteps, context, mask, skip_layers=(1,))
        without_skip = model(x, timesteps, context, mask)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(actual, without_skip)


def test_forward_without_mask_is_full_length():
    reference, model = _make_models(False)
    x, timesteps, context, mask = _make_inputs()
    full_mask = torch.ones_like(mask)
    with torch.no_grad():
        expected = reference(x, timesteps, context, full_mask)
        actual = model(x, timesteps, context, None)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_gradient_checkpointing_matches_plain_forward():
    _, model = _make_models(False)
    x, timesteps, context, mask = _make_inputs()
    with torch.no_grad():
        expected = model(x, timesteps, context, mask)
    model.train()
    model.enable_gradient_checkpointing()
    x_grad = x.clone().requires_grad_(True)
    out = model(x_grad, timesteps, context, mask)
    out.sum().backward()
    torch.testing.assert_close(out.detach(), expected, atol=1e-5, rtol=1e-5)
    assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all()


def test_partial_gradient_checkpointing_skips_last_blocks():
    _, model = _make_models(False)
    x, timesteps, context, mask = _make_inputs()
    model.train()

    # reference: every block is checkpointed
    model.enable_gradient_checkpointing()
    x_ref = x.clone().requires_grad_(True)
    model(x_ref, timesteps, context, mask).sum().backward()

    model.enable_gradient_checkpointing(skip_blocks=2)
    assert [block.gradient_checkpointing for block in model.layers] == [True, False, False]
    assert all(block.gradient_checkpointing for block in model.text_embedder.blocks)
    x_grad = x.clone().requires_grad_(True)
    out = model(x_grad, timesteps, context, mask)
    out.sum().backward()
    torch.testing.assert_close(x_grad.grad, x_ref.grad, atol=1e-5, rtol=1e-5)

    # re-applying with fewer skipped blocks re-enables checkpointing on the previously skipped blocks
    model.enable_gradient_checkpointing(skip_blocks=0)
    assert all(block.gradient_checkpointing for block in model.layers)
    model.enable_gradient_checkpointing(skip_blocks=3)
    assert not any(block.gradient_checkpointing for block in model.layers)

    with pytest.raises(ValueError):
        model.enable_gradient_checkpointing(skip_blocks=4)
    with pytest.raises(ValueError):
        model.enable_gradient_checkpointing(skip_blocks=-1)
