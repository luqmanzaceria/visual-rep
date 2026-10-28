"""Building blocks shared by the predictor and the probe."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Map coordinates (…, P) in roughly [-1, 1] to [x, sin(2^k pi x), cos(2^k pi x)]."""

    def __init__(self, in_dim: int, n_freq: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.n_freq = n_freq
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi)

    @property
    def out_dim(self) -> int:
        return self.in_dim * (1 + 2 * self.n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x[..., None] * self.freqs  # (…, P, F)
        return torch.cat([x, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)


class TokenEmbedding(nn.Module):
    """feat (B,N,D) + pos (B,N,P) -> (B,N,d_model)."""

    def __init__(self, feat_dim: int, pos_dim: int, d_model: int, pos_range: float = 1.0, n_freq: int = 6):
        super().__init__()
        self.pos_range = pos_range
        self.fourier = FourierFeatures(pos_dim, n_freq)
        self.feat_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d_model))
        self.pos_proj = nn.Sequential(nn.Linear(self.fourier.out_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, feat: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.feat_proj(feat) + self.pos_proj(self.fourier(pos / self.pos_range))


def make_encoder(d_model: int, n_layers: int, n_heads: int, dropout: float = 0.0) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
