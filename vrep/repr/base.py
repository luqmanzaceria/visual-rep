"""The common currency between representations and the predictor.

Every representation turns an observation into a *set of tokens*. A token has a
feature vector and a position. For image-aligned features the position is the
patch centre in normalised image coordinates; for Gaussians it is the 3D centre.
The predictor only ever sees ``TokenBatch`` objects, so it is literally the same
module (same architecture, same hyper-parameters) for all representations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass
class TokenBatch:
    feat: torch.Tensor  # (B, N, D)
    pos: torch.Tensor  # (B, N, P)   P = 2 (image) or 3 (world)
    mask: Optional[torch.Tensor] = None  # (B, N) True = valid token

    @property
    def B(self) -> int:
        return self.feat.shape[0]

    @property
    def N(self) -> int:
        return self.feat.shape[1]

    @property
    def D(self) -> int:
        return self.feat.shape[2]

    def to(self, device) -> "TokenBatch":
        return TokenBatch(
            self.feat.to(device), self.pos.to(device), None if self.mask is None else self.mask.to(device)
        )

    def detach(self) -> "TokenBatch":
        return TokenBatch(self.feat.detach(), self.pos.detach(), self.mask)

    def with_feat(self, feat: torch.Tensor) -> "TokenBatch":
        return replace(self, feat=feat)

    def with_pos(self, pos: torch.Tensor) -> "TokenBatch":
        return replace(self, pos=pos)

    def __getitem__(self, idx) -> "TokenBatch":
        return TokenBatch(self.feat[idx], self.pos[idx], None if self.mask is None else self.mask[idx])


class Representation:
    """Interface implemented by each of the three representations.

    * ``feat_dim`` / ``pos_dim``: token layout the predictor must accept.
    * ``out_dim``: what the predictor outputs per token (next features for
      image-aligned representations, a 3D displacement for Gaussians).
    * ``apply_prediction``: turn a predictor output into the next ``TokenBatch``.
    """

    name: str = "base"
    feat_dim: int
    pos_dim: int
    out_dim: int
    residual: bool  # predictor output is added to the input features (2D) or to positions (3D)

    def apply_prediction(self, tokens: TokenBatch, out: torch.Tensor) -> TokenBatch:
        raise NotImplementedError

    def target(self, next_tokens: TokenBatch, tokens: TokenBatch) -> torch.Tensor:
        """The regression target for the predictor given the true next tokens."""
        raise NotImplementedError
