"""Standalone text-only Qwen3-VL tower for MiniMax-H3 conditioning.

The execution order and numerics mirror ComfyUI's Apache-2.0 Qwen/Llama text
runtime, but this module has no ComfyUI dependency.  It intentionally implements
only the truncated language tower needed by image-only H3 training and preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MiniMaxQwen3VLTextConfig:
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = 50
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    rope_dims: tuple[int, int, int] = (24, 20, 20)
    interleaved_mrope: bool = True
    attention_bias: bool = False
    tie_word_embeddings: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, *, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), weight, self.eps)


def _precompute_freqs_cis(
    head_dim: int,
    position_ids: torch.Tensor,
    theta: float,
    rope_dims: tuple[int, int, int],
    *,
    interleaved_mrope: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerator = torch.arange(0, head_dim, 2, device=position_ids.device).float()
    inv_freq = 1.0 / (theta ** (numerator / head_dim))
    expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    positions = position_ids[:, None, :].float()
    freqs = (expanded @ positions).transpose(1, 2)
    if position_ids.shape[0] > 1 and interleaved_mrope:
        interleaved = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            interleaved[..., slice(offset, rope_dims[axis] * 3, 3)] = freqs[
                axis, ..., slice(offset, rope_dims[axis] * 3, 3)
            ]
        embedding = torch.cat((interleaved, interleaved), dim=-1)
        cosine = embedding.cos().unsqueeze(0)
        sine = embedding.sin().unsqueeze(0)
    else:
        embedding = torch.cat((freqs, freqs), dim=-1)
        cosine = embedding.cos().unsqueeze(1)
        sine = embedding.sin().unsqueeze(1)
    split = sine.shape[-1] // 2
    return cosine, sine[..., :split], -sine[..., split:]


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, sine, negative_sine = frequencies
    query_output = query * cosine
    query_split = query_output.shape[-1] // 2
    query_output[..., :query_split].addcmul_(query[..., query_split:], negative_sine)
    query_output[..., query_split:].addcmul_(query[..., :query_split], sine)
    key_output = key * cosine
    key_split = key_output.shape[-1] // 2
    key_output[..., :key_split].addcmul_(key[..., key_split:], negative_sine)
    key_output[..., key_split:].addcmul_(key[..., :key_split], sine)
    return query_output.to(query.dtype), key_output.to(key.dtype)


class Attention(nn.Module):
    def __init__(self, config: MiniMaxQwen3VLTextConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.inner_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, sequence, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, sequence, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rope(query, key, frequencies)
        mask = attention_mask
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=self.num_heads != self.num_key_value_heads,
        )
        output = output.transpose(1, 2).reshape(batch, sequence, self.inner_size)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: MiniMaxQwen3VLTextConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderLayer(nn.Module):
    def __init__(self, config: MiniMaxQwen3VLTextConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype)
        self.mlp = MLP(config, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask, frequencies):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), attention_mask, frequencies)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class MiniMaxQwen3VLTextModel(nn.Module):
    def __init__(self, config: MiniMaxQwen3VLTextConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            DecoderLayer(config, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)
        )
        self.norm = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden_states = self.embed_tokens(input_ids)
        sequence = hidden_states.shape[1]
        positions = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
        frequencies = _precompute_freqs_cis(
            self.config.head_dim,
            positions,
            self.config.rope_theta,
            self.config.rope_dims,
            interleaved_mrope=self.config.interleaved_mrope,
        )
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(hidden_states.dtype).reshape(
                attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
            ).expand(attention_mask.shape[0], 1, sequence, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(hidden_states.dtype).min / 4)
        if sequence > 1:
            causal = torch.empty(sequence, sequence, dtype=hidden_states.dtype, device=hidden_states.device)
            causal.fill_(torch.finfo(hidden_states.dtype).min / 4).triu_(1)
            mask = causal if mask is None else mask + causal
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, frequencies)
        return SimpleNamespace(last_hidden_state=hidden_states)
