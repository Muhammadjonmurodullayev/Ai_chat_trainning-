"""Multi-Head Attention with RoPE support + KV-cache."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .rope import apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None, rope_cos=None, rope_sin=None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            Q, K = apply_rotary_pos_emb(Q, K, rope_cos, rope_sin)

        new_kv = None
        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)
            new_kv = (K, V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self._scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        return self.W_o(out), new_kv


def create_causal_mask(seq_len: int, device=None):
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)


def create_padding_mask(input_ids: torch.Tensor, pad_id: int = 0):
    return (input_ids != pad_id).unsqueeze(1).unsqueeze(2).float()
