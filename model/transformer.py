"""
MiniTransformer v2 — GPT-like decoder-only transformer.
RoPE + SwiGLU + Residual Scaling + Weight Tying + KV-Cache.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .embeddings import TokenEmbedding
from .rope import RotaryPositionalEncoding
from .attention import MultiHeadAttention, create_causal_mask, create_padding_mask
from .feedforward import SwiGLUFeedForward


@dataclass
class TransformerConfig:
    vocab_size: int = 5000
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 5
    ff_dim: int = 768
    max_seq_len: int = 256
    dropout: float = 0.2
    padding_idx: int = 0


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attention = MultiHeadAttention(config.embed_dim, config.num_heads, config.dropout)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = SwiGLUFeedForward(config.embed_dim, config.ff_dim, config.dropout)
        self._res_scale = 1.0 / math.sqrt(2 * config.num_layers)

    def forward(self, x, mask=None, rope_cos=None, rope_sin=None, kv_cache=None):
        normed = self.ln1(x)
        attn_out, new_kv = self.attention(normed, normed, normed, mask, rope_cos, rope_sin, kv_cache)
        x = x + attn_out * self._res_scale
        x = x + self.ffn(self.ln2(x)) * self._res_scale
        return x, new_kv


class MiniTransformer(nn.Module):
    """GPT-like decoder-only transformer v2."""

    def __init__(self, config: TransformerConfig = None):
        super().__init__()
        self.config = config or TransformerConfig()
        c = self.config

        self.token_embedding = TokenEmbedding(c.vocab_size, c.embed_dim, c.padding_idx)
        self.rope = RotaryPositionalEncoding(c.embed_dim // c.num_heads, c.max_seq_len)
        self.embed_dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(c, i) for i in range(c.num_layers)])
        self.ln_final = nn.LayerNorm(c.embed_dim)
        self.output_head = nn.Linear(c.embed_dim, c.vocab_size, bias=False)

        self._init_weights()
        self.output_head.weight = self.token_embedding.embedding.weight  # weight tying

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "w_gate" in name or "w_up" in name or "w_down" in name:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def forward(self, input_ids, mask=None, use_kv_cache=False, kv_caches=None):
        seq_len = input_ids.size(1)
        if mask is None:
            causal = create_causal_mask(seq_len, device=input_ids.device)
            pad = create_padding_mask(input_ids, self.config.padding_idx)
            mask = causal * pad

        x = self.embed_dropout(self.token_embedding(input_ids))
        rope_cos, rope_sin = self.rope(x, seq_len)

        new_caches = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches else None
            x, new_kv = block(x, mask, rope_cos, rope_sin, kv)
            if use_kv_cache:
                new_caches.append(new_kv)

        logits = self.output_head(self.ln_final(x))
        return (logits, new_caches) if use_kv_cache else logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> dict:
        return {
            "architecture": "MiniTransformer v2 (RoPE + SwiGLU)",
            "vocab_size": self.config.vocab_size,
            "embed_dim": self.config.embed_dim,
            "num_heads": self.config.num_heads,
            "num_layers": self.config.num_layers,
            "ff_dim": self.config.ff_dim,
            "max_seq_len": self.config.max_seq_len,
            "parameters": f"{self.count_parameters():,}",
            "parameters_M": f"{self.count_parameters() / 1e6:.2f}M",
        }
