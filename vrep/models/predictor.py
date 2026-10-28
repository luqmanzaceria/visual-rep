"""The action-conditioned predictor shared by all representations.

    tokens_t (features + positions), action_t  ->  per-token output

* For image-aligned representations the output is a residual on the features
  (next features = features + output).
* For Gaussians the output is a displacement of each centre.

Optional mechanisms used in the "give DINO the missing property" experiment:

* ``n_memory > 0``: learned memory tokens that are carried across rollout steps
  (a cheap route to *persistence*: the model can stash what it saw).
* ``gated=True``: every token's update is multiplied by a sigmoid gate; an L1
  penalty on the gates (applied by the training loop) encourages the model to
  change only the tokens that the action affects (*locality*).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from vrep.models.common import TokenEmbedding, make_encoder
from vrep.repr.base import TokenBatch


@dataclass
class PredictorConfig:
    feat_dim: int
    pos_dim: int
    out_dim: int
    action_dim: int = 2
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_freq: int = 6
    pos_range: float = 1.0  # positions are divided by this before Fourier encoding
    action_scale: float = 0.06  # actions are divided by this (max action)
    n_memory: int = 0
    gated: bool = False
    dropout: float = 0.0
    out_scale: float = 1.0  # multiplies the head output (keeps init small for residual targets)


@dataclass
class PredictorOutput:
    out: torch.Tensor  # (B,N,out_dim)
    gate: Optional[torch.Tensor] = None  # (B,N)
    memory: Optional[torch.Tensor] = None  # (B,M,d_model)


class Predictor(nn.Module):
    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = TokenEmbedding(cfg.feat_dim, cfg.pos_dim, d, cfg.pos_range, cfg.n_freq)
        self.action_mlp = nn.Sequential(nn.Linear(cfg.action_dim, d), nn.GELU(), nn.Linear(d, d))
        self.action_token = nn.Parameter(torch.zeros(1, 1, d))
        self.memory_init = nn.Parameter(torch.randn(1, cfg.n_memory, d) * 0.02) if cfg.n_memory > 0 else None
        self.encoder = make_encoder(d, cfg.n_layers, cfg.n_heads, cfg.dropout)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gate = nn.Linear(d, 1) if cfg.gated else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 2.0)  # gates start open (sigmoid(2) ~ 0.88)

    @property
    def n_memory(self) -> int:
        return self.cfg.n_memory

    def initial_memory(self, B: int) -> Optional[torch.Tensor]:
        if self.memory_init is None:
            return None
        return self.memory_init.expand(B, -1, -1)

    def forward(self, tokens: TokenBatch, action: torch.Tensor, memory: Optional[torch.Tensor] = None) -> PredictorOutput:
        B, N = tokens.B, tokens.N
        x = self.embed(tokens.feat, tokens.pos)
        a = self.action_mlp(action / self.cfg.action_scale)  # (B,d)
        x = x + a[:, None, :]
        seq = [self.action_token.expand(B, -1, -1) + a[:, None, :]]
        n_prefix = 1
        if self.memory_init is not None:
            mem = memory if memory is not None else self.initial_memory(B)
            seq.append(mem)
            n_prefix += mem.shape[1]
        seq.append(x)
        h = torch.cat(seq, dim=1)
        pad = None
        if tokens.mask is not None:
            pad = torch.cat([torch.zeros(B, n_prefix, dtype=torch.bool, device=h.device), ~tokens.mask], dim=1)
        y = self.encoder(h, src_key_padding_mask=pad)
        y_tok = self.out_norm(y[:, n_prefix:])
        out = self.head(y_tok) * self.cfg.out_scale
        gate = None
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(y_tok)).squeeze(-1)
            out = out * gate[..., None]
        if tokens.mask is not None:
            out = out * tokens.mask[..., None]
        new_mem = y[:, 1:n_prefix] if self.memory_init is not None else None
        return PredictorOutput(out=out, gate=gate, memory=new_mem)
