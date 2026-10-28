"""Open-loop rollouts of the predictor (used by training, evaluation and planning)."""

from __future__ import annotations

from typing import Optional

import torch

from vrep.models.predictor import Predictor, PredictorOutput
from vrep.repr.base import TokenBatch
from vrep.repr.gaussians import GaussianRepresentation, GaussianScene


def rollout_tokens(
    predictor: Predictor, rep, tokens: TokenBatch, actions: torch.Tensor, memory: Optional[torch.Tensor] = None
) -> tuple[list[TokenBatch], list[PredictorOutput]]:
    """Roll the predictor forward for actions (B,H,2). Returns the H predicted TokenBatches (t+1..t+H)."""
    preds, outs = [], []
    mem = memory if memory is not None else predictor.initial_memory(tokens.B)
    for k in range(actions.shape[1]):
        out = predictor(tokens, actions[:, k], mem)
        mem = out.memory
        tokens = rep.apply_prediction(tokens, out.out)
        preds.append(tokens)
        outs.append(out)
    return preds, outs


def rollout_scene(
    predictor: Predictor, rep: GaussianRepresentation, scene: GaussianScene, actions: torch.Tensor, memory: Optional[torch.Tensor] = None
) -> tuple[list[GaussianScene], list[PredictorOutput]]:
    tokens = rep.tokens(scene)
    preds, outs = rollout_tokens(predictor, rep, tokens, actions, memory)
    return [scene.with_mu(p.pos) for p in preds], outs
