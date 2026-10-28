"""Metrics for scene-changing interaction with granular material.

The common currency is an occupancy map: the number of grains in each cell of
a ``grid x grid`` partition of the table (normalised by ``OCC_NORM``). For an
open-loop rollout from t0 with the recorded actions, every metric compares the
*change* of the map, ``occ_k - occ_0``, between the model and the simulator:

* ``change_err``      mean |pred change - true change| over all cells,
* ``changed_err``     the same over cells that really changed (did the model dig?),
* ``spurious``        |pred change| over cells far from the pusher's path that did not
                      change (locality: the rest of the pile must stay put),
* ``kept_ratio``      on cells that changed, |pred change| / |true change| at the end of
                      the rollout, after the pusher has stopped (permanence: 1 = the trench
                      is kept, 0 = the model forgot it or never made it).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

OCC_NORM = 8.0  # grains per cell that map to 1.0


def grain_occupancy(grain_pos: np.ndarray, table_half: float, grid: int = 8) -> np.ndarray:
    """grain_pos (..., G, 3) -> normalised counts (..., grid*grid), row-major over (y, x)."""
    gp = np.asarray(grain_pos)
    lead = gp.shape[:-2]
    flat = gp.reshape(-1, gp.shape[-2], 3)
    out = np.zeros((flat.shape[0], grid * grid), np.float32)
    for b in range(flat.shape[0]):
        xy = flat[b, :, :2]
        ij = np.floor((xy + table_half) / (2 * table_half) * grid).astype(int)
        ok = np.all((ij >= 0) & (ij < grid), axis=1)
        np.add.at(out[b], ij[ok, 1] * grid + ij[ok, 0], 1.0)
    return (out / OCC_NORM).reshape(*lead, grid * grid)


def cell_centres(table_half: float, grid: int) -> np.ndarray:
    c = (np.arange(grid) + 0.5) / grid * 2 * table_half - table_half
    y, x = np.meshgrid(c, c, indexing="ij")
    return np.stack([x.reshape(-1), y.reshape(-1)], -1)


def far_from_path(pusher_xy: np.ndarray, table_half: float, grid: int, dist: float) -> np.ndarray:
    """(grid*grid,) True for cells whose centre is farther than ``dist`` from every pusher position."""
    cc = cell_centres(table_half, grid)
    d = np.linalg.norm(cc[:, None, :] - pusher_xy[None, :, :2], axis=-1).min(1)
    return d > dist


def explicit_occupancy(scene, table_half: float, grid: int, n_grains: int) -> torch.Tensor:
    """Occupancy from the grain-group Gaussians of a GaussianScene: (B, grid*grid), rescaled to grain counts."""
    from vrep.repr.gaussians import GROUP_GRAIN

    m = (scene.group == GROUP_GRAIN) & scene.valid
    xy = scene.mu[..., :2]
    ij = torch.floor((xy + table_half) / (2 * table_half) * grid).long().clamp(0, grid - 1)
    idx = ij[..., 1] * grid + ij[..., 0]
    out = torch.zeros(scene.B, grid * grid, device=scene.mu.device)
    out.scatter_add_(1, idx, m.float())
    return out / m.float().sum(1, keepdim=True).clamp_min(1) * n_grains / OCC_NORM


@torch.no_grad()
def granular_metrics(exp, episodes: list[int], cams: list[int], probe=None, horizon: Optional[int] = None, max_items: int = 300,
                     far_dist: float = 0.10, changed_thresh: float = 0.75 / OCC_NORM) -> dict:
    from vrep.training.rollout import rollout_scene, rollout_tokens

    store = exp.store
    scene_cfg = store.scene
    probe = probe or exp.probe
    grid = probe.grid
    H = horizon or store.T
    th = scene_cfg.table_half
    items = [(e, 0, c) for e in episodes if store[e].kind == "plow" for c in cams][:max_items]
    if not items:
        return {"n_rollouts": 0}
    rows = {"probe": [], "explicit": []}
    for i in range(0, len(items), 16):
        chunk = items[i : i + 16]
        actions = torch.from_numpy(np.stack([store[e].actions[t : t + H] for (e, t, c) in chunk])).float().to(exp.device)
        if exp.is_gauss:
            scene = exp.scene_items(chunk)
            tok0 = exp.rep.tokens(scene)
            scenes, _ = rollout_scene(exp.predictor, exp.rep, scene, actions)
            preds = [exp.rep.tokens(s) for s in scenes]
            ex0 = explicit_occupancy(scene, th, grid, scene_cfg.n_grains)
            ex = torch.stack([explicit_occupancy(s, th, grid, scene_cfg.n_grains) for s in scenes], 1)
            ex_change = (ex - ex0[:, None]).cpu().numpy()
        else:
            tok0 = exp.encode_items(chunk)
            preds, _ = rollout_tokens(exp.predictor, exp.rep, tok0, actions)
            ex_change = None
        ref = probe(tok0)[0]
        pr_change = (torch.stack([probe(p)[0] for p in preds], 1) - ref[:, None]).cpu().numpy()  # (B,H,cells)
        for b, (e, t, c) in enumerate(chunk):
            ep = store[e]
            occ = grain_occupancy(ep.grain_pos[t : t + H + 1], th, grid)
            gt_change = occ[1:] - occ[:1]  # (H, cells)
            changed = np.abs(gt_change[-1]) > changed_thresh
            far = far_from_path(ep.pusher_pos[t : t + H + 1], th, grid, far_dist) & (np.abs(gt_change).max(0) < 1e-6)
            for name, ch in (("probe", pr_change[b]), ("explicit", None if ex_change is None else ex_change[b])):
                if ch is None:
                    continue
                kept = np.abs(ch[-1][changed]).sum() / max(np.abs(gt_change[-1][changed]).sum(), 1e-6) if changed.any() else np.nan
                rows[name].append({
                    "change_err": np.abs(ch - gt_change).mean(1),  # (H,)
                    "changed_err": np.abs(ch[-1] - gt_change[-1])[changed].mean() if changed.any() else np.nan,
                    "spurious": np.abs(ch[-1])[far].mean() if far.any() else np.nan,
                    "kept_ratio": kept,
                })
    out = {"n_rollouts": len(items), "horizon": H}
    for name, rs in rows.items():
        if not rs:
            continue
        ce = np.stack([r["change_err"] for r in rs])
        for k in (1, 4, 8, H):
            if k <= H:
                out[f"{name}/k{k}/change_err"] = {"mean": float(ce[:, k - 1].mean()) * OCC_NORM, "n": len(rs)}
        for key in ("changed_err", "spurious", "kept_ratio"):
            v = np.array([r[key] for r in rs], float)
            v = v[np.isfinite(v)]
            scale = 1.0 if key == "kept_ratio" else OCC_NORM
            out[f"{name}/{key}"] = {"mean": float(v.mean()) * scale if v.size else float("nan"), "median": float(np.median(v)) * scale if v.size else float("nan"), "n": int(v.size)}
    return out
