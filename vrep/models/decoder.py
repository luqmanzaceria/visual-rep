"""A small convolutional decoder from image-aligned tokens back to RGB.

Only used for *visualising* what the predicted tokens of the 2D representations
encode (the Gaussian representation renders itself). Trained on real tokens,
applied to predicted ones.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vrep.repr.base import TokenBatch


class TokenDecoder(nn.Module):
    def __init__(self, feat_dim: int, grid: int, out_size: int = 64, width: int = 128):
        super().__init__()
        self.grid = grid
        self.out_size = out_size
        n_up = int(round(math.log2(out_size / grid)))
        assert grid * 2**n_up == out_size, "out_size must be grid * 2^k"
        self.inp = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, width))
        layers = []
        c = width
        for i in range(n_up):
            c_out = max(32, c // 2)
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c_out, 3, padding=1), nn.GELU(), nn.Conv2d(c_out, c_out, 3, padding=1), nn.GELU()]
            c = c_out
        layers += [nn.Conv2d(c, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: TokenBatch) -> torch.Tensor:
        """-> (B, out_size, out_size, 3), roughly in [0, 1] (clamp for display)."""
        B, N, _ = tokens.feat.shape
        x = self.inp(tokens.feat).transpose(1, 2).reshape(B, -1, self.grid, self.grid)
        return self.net(x).permute(0, 2, 3, 1) + 0.5
