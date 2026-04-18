"""Rotary Position Encoding (RoPE) — Su et al., 2021."""

import torch
import torch.nn as nn


class RotaryPositionalEncoding(nn.Module):
    """Precomputes cos/sin tables for RoPE, applied to Q and K in attention."""

    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> tuple:
        seq_len = seq_len or x.size(1)
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len, :], self.sin_cached[:, :seq_len, :]


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to Q and K: [batch, heads, seq_len, head_dim]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
