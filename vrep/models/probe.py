"""Readout from tokens to object positions.

Used to compare representations in a common currency (metres on the table):
K learned object queries cross-attend to the tokens and output each object's
(x, y) position in world frame plus a visibility logit. It is trained on real
(encoded) tokens and then applied to *predicted* tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vrep.models.common import TokenEmbedding, make_encoder
from vrep.repr.base import TokenBatch


class ObjectProbe(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        pos_dim: int,
        n_objects: int,
        d_model: int = 192,
        n_layers: int = 2,
        n_heads: int = 4,
        pos_range: float = 1.0,
        n_freq: int = 6,
        out_scale: float = 0.3,
    ):
        super().__init__()
        self.n_objects = n_objects
        self.out_scale = out_scale
        self.embed = TokenEmbedding(feat_dim, pos_dim, d_model, pos_range, n_freq)
        self.encoder = make_encoder(d_model, n_layers, n_heads)
        self.queries = nn.Parameter(torch.randn(1, n_objects, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 3)  # x, y, visibility logit

    def forward(self, tokens: TokenBatch) -> tuple[torch.Tensor, torch.Tensor]:
        B = tokens.B
        x = self.embed(tokens.feat, tokens.pos)
        pad = None if tokens.mask is None else ~tokens.mask
        mem = self.encoder(x, src_key_padding_mask=pad)
        q = self.queries.expand(B, -1, -1)
        y = self.decoder(q, mem, memory_key_padding_mask=pad)
        o = self.head(self.norm(y))
        return o[..., :2] * self.out_scale, o[..., 2]


class GridProbe(nn.Module):
    """Readout from tokens to a coarse occupancy map of granular material.

    One learned query per table cell cross-attends to the tokens and predicts the
    (normalised) number of grains in that cell. Like ``ObjectProbe`` it is trained
    on real tokens and applied to predicted ones.
    """

    def __init__(self, feat_dim: int, pos_dim: int, grid: int = 8, d_model: int = 192, n_layers: int = 2, n_heads: int = 4,
                 pos_range: float = 1.0, n_freq: int = 6):
        super().__init__()
        self.grid = grid
        self.embed = TokenEmbedding(feat_dim, pos_dim, d_model, pos_range, n_freq)
        self.encoder = make_encoder(d_model, n_layers, n_heads)
        self.queries = nn.Parameter(torch.randn(1, grid * grid, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tokens: TokenBatch) -> tuple[torch.Tensor, None]:
        B = tokens.B
        x = self.embed(tokens.feat, tokens.pos)
        pad = None if tokens.mask is None else ~tokens.mask
        mem = self.encoder(x, src_key_padding_mask=pad)
        y = self.decoder(self.queries.expand(B, -1, -1), mem, memory_key_padding_mask=pad)
        return self.head(self.norm(y))[..., 0], None
