"""Cross-entropy method over action sequences."""

from __future__ import annotations

from typing import Callable

import numpy as np


def cem_plan(
    cost_fn: Callable[[np.ndarray], np.ndarray],
    horizon: int,
    action_dim: int,
    bound: float,
    rng: np.random.Generator,
    n_samples: int = 64,
    n_elite: int = 8,
    n_iters: int = 3,
    init_mean: np.ndarray | None = None,
    init_std: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return the best action sequence (horizon, action_dim) and its cost.

    ``cost_fn`` maps candidate sequences (S, horizon, action_dim) to costs (S,).
    """
    mean = np.zeros((horizon, action_dim)) if init_mean is None else init_mean.copy()
    std = np.full((horizon, action_dim), bound * 0.7 if init_std is None else init_std)
    best_seq, best_cost = mean.copy(), np.inf
    for it in range(n_iters):
        samples = rng.normal(mean, std, size=(n_samples, horizon, action_dim))
        samples = np.clip(samples, -bound, bound)
        if it > 0:
            samples[0] = best_seq  # keep the elite
        costs = np.asarray(cost_fn(samples))
        order = np.argsort(costs)
        elite = samples[order[:n_elite]]
        if costs[order[0]] < best_cost:
            best_cost, best_seq = float(costs[order[0]]), samples[order[0]].copy()
        mean = elite.mean(0)
        std = elite.std(0) + 1e-3
    return best_seq, best_cost
