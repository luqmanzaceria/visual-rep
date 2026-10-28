"""The three property metrics (persistence, viewpoint consistency, locality).

All metrics are expressed in a common currency: *object positions in metres on
the table*, read out from the predicted tokens with a probe (and, for Gaussians,
also directly from the centroids of each object's Gaussians).

For an open-loop rollout from frame t0 with the recorded actions we compute, for
every step k and object j:

* ``abs_err``  = | readout(pred_k)[j] - gt_{t0+k}[j] |
* ``disp_err`` = | (readout(pred_k)[j] - readout(real_t0)[j]) - (gt_{t0+k}[j] - gt_t0[j]) |

``disp_err`` cancels the readout's constant bias at t0 and is the quantity we
report: it asks "did the model move object j by the right amount?".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from vrep.training.builder import Experiment
from vrep.training.rollout import rollout_scene, rollout_tokens


@dataclass
class RolloutResult:
    pred_xy: np.ndarray  # (B,H,K,2) readout from predicted tokens
    ref_xy: np.ndarray  # (B,K,2) readout from real tokens at t0
    gt_xy: np.ndarray  # (B,H+1,K,2) ground truth at t0..t0+H
    explicit_xy: Optional[np.ndarray]  # (B,H+1,K,2) Gaussian centroids (gauss only), index 0 = t0
    items: list[tuple[int, int, int]]

    @property
    def disp_err(self) -> np.ndarray:  # (B,H,K)
        pred_disp = self.pred_xy - self.ref_xy[:, None]
        gt_disp = self.gt_xy[:, 1:] - self.gt_xy[:, :1]
        return np.linalg.norm(pred_disp - gt_disp, axis=-1)

    @property
    def abs_err(self) -> np.ndarray:
        return np.linalg.norm(self.pred_xy - self.gt_xy[:, 1:], axis=-1)

    @property
    def pred_disp_norm(self) -> np.ndarray:  # (B,H,K) how much the model *thinks* each object moved
        return np.linalg.norm(self.pred_xy - self.ref_xy[:, None], axis=-1)

    @property
    def gt_disp_norm(self) -> np.ndarray:
        return np.linalg.norm(self.gt_xy[:, 1:] - self.gt_xy[:, :1], axis=-1)

    @property
    def explicit_disp_err(self) -> Optional[np.ndarray]:
        if self.explicit_xy is None:
            return None
        pred_disp = self.explicit_xy[:, 1:] - self.explicit_xy[:, :1]
        gt_disp = self.gt_xy[:, 1:] - self.gt_xy[:, :1]
        return np.linalg.norm(pred_disp - gt_disp, axis=-1)

    @property
    def explicit_pred_disp_norm(self) -> Optional[np.ndarray]:
        if self.explicit_xy is None:
            return None
        return np.linalg.norm(self.explicit_xy[:, 1:] - self.explicit_xy[:, :1], axis=-1)


@torch.no_grad()
def rollout_positions(exp: Experiment, items: list[tuple[int, int, int]], horizon: int, probe=None, batch_size: int = 32,
                      rng: Optional[np.random.Generator] = None) -> RolloutResult:
    """Open-loop rollouts with the recorded actions, read out as object positions."""
    probe = probe or exp.probe
    rng = rng or np.random.default_rng(0)
    exp.predictor.eval()
    probe.eval()
    K = exp.store.n_objects
    pred_all, ref_all, gt_all, expl_all = [], [], [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        actions = torch.from_numpy(np.stack([exp.store[e].actions[t : t + horizon] for (e, t, c) in chunk])).float().to(exp.device)
        gt = np.stack([exp.store[e].obj_pos[t : t + horizon + 1, :, :2] for (e, t, c) in chunk])
        if exp.is_gauss:
            scene = exp.scene_items(chunk, rng)
            tokens0 = exp.rep.tokens(scene)
            scenes, _ = rollout_scene(exp.predictor, exp.rep, scene, actions)
            preds = [exp.rep.tokens(s) for s in scenes]
            expl = torch.stack([scene.centroids(K)] + [s.centroids(K) for s in scenes], 1)[..., :2]
            expl_all.append(expl.cpu().numpy())
        else:
            tokens0 = exp.encode_items(chunk)
            preds, _ = rollout_tokens(exp.predictor, exp.rep, tokens0, actions)
        ref = probe(tokens0)[0]
        pred = torch.stack([probe(p)[0] for p in preds], 1)
        pred_all.append(pred.cpu().numpy())
        ref_all.append(ref.cpu().numpy())
        gt_all.append(gt)
    return RolloutResult(
        np.concatenate(pred_all), np.concatenate(ref_all), np.concatenate(gt_all),
        np.concatenate(expl_all) if expl_all else None, items,
    )


def _summ(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "n": int(x.size)}


def _visible_throughout(ep, cam: int, k: int, t0: int, t1: int, frac: float = 0.6) -> bool:
    v = ep.visibility[t0 : t1 + 1, cam, k]
    return v.min() >= frac * max(ep.visibility[:, cam, k].max(), 1)


# ------------------------------------------------------------------ locality
def locality_metrics(exp: Experiment, episodes: list[int], cams: list[int], horizon: int = 8, probe=None, ks=(1, 2, 4, 8),
                     moved_thresh: float = 0.02, static_thresh: float = 0.01, max_items: int = 400) -> dict:
    """On 'push' episodes: did the model move the pushed object, and only that object?"""
    store = exp.store
    eps = [e for e in episodes if store[e].kind == "push"]
    items = [(e, 0, c) for e in eps for c in cams][:max_items]
    if not items:
        return {}
    res = rollout_positions(exp, items, horizon, probe)
    out = {"n_rollouts": len(items)}
    for name, disp_err, pred_disp in [("probe", res.disp_err, res.pred_disp_norm), ("explicit", res.explicit_disp_err, res.explicit_pred_disp_norm)]:
        if disp_err is None:
            continue
        for k in ks:
            if k > horizon:
                continue
            tgt_err, spurious, tgt_disp = [], [], []
            for b, (e, t, c) in enumerate(items):
                ep = store[e]
                gt_disp = np.linalg.norm(ep.obj_pos[t + k, :, :2] - ep.obj_pos[t, :, :2], axis=-1)
                for j in range(store.n_objects):
                    if j == ep.target and gt_disp[j] > moved_thresh:
                        tgt_err.append(disp_err[b, k - 1, j])
                        tgt_disp.append(gt_disp[j])
                    elif j != ep.target and gt_disp[j] < static_thresh:
                        spurious.append(pred_disp[b, k - 1, j])
            out[f"{name}/k{k}/target_disp_err_cm"] = _summ(np.array(tgt_err) * 100)
            out[f"{name}/k{k}/spurious_motion_cm"] = _summ(np.array(spurious) * 100)
            out[f"{name}/k{k}/gt_target_disp_cm"] = _summ(np.array(tgt_disp) * 100)
    return out


# --------------------------------------------------------------- persistence
def find_occlusion(ep, cam: int, k: int, low: float = 0.3, high: float = 0.7) -> Optional[tuple[int, int, int]]:
    """Occlusion event of object k in camera cam: (t_vis, t_occ, t_re) or None.

    t_vis: last fully visible frame before the occlusion (the rollout starts here),
    t_occ: first frame in which the object is (nearly) fully hidden,
    t_re:  first frame after t_occ in which it is fully visible again.
    The pusher takes a few frames to slide across an object, so t_vis < t_occ - 1 is common.
    """
    v = ep.visibility[:, cam, k].astype(float)
    vmax = v.max()
    if vmax < 20:
        return None
    occ = np.nonzero(v < low * vmax)[0]
    if occ.size == 0:
        return None
    t_occ = int(occ[0])
    vis_before = np.nonzero(v[:t_occ] >= high * vmax)[0]
    if vis_before.size == 0:
        return None  # hidden from the start
    t_vis = int(vis_before[-1])
    re = np.nonzero(v[t_occ:] >= high * vmax)[0]
    if re.size == 0:
        return None
    return t_vis, t_occ, t_occ + int(re[0])


def persistence_metrics(exp: Experiment, episodes: list[int], cams: Optional[list[int]] = None, probe=None, extra: int = 2,
                        control_episodes: Optional[list[int]] = None, max_items: int = 400) -> dict:
    """On 'sweep' episodes: roll out through the occlusion; error at reappearance.

    The control is the same measurement for objects that stay visible (and
    static) over rollouts of the same length, taken from non-sweep episodes.
    """
    store = exp.store
    cases = []  # (e, t0, cam, k_target, t_occ, t_re)
    for e in episodes:
        ep = store[e]
        if ep.kind != "sweep":
            continue
        cam_list = [ep.cam] if cams is None else [c for c in cams if c == ep.cam]
        for c in cam_list:
            occ = find_occlusion(ep, c, ep.target)
            if occ is None:
                continue
            t0, t_occ, t_re = occ
            if t_re + extra > store.T:
                continue
            cases.append((e, t0, c, ep.target, t_occ, t_re))
    cases = cases[:max_items]
    out = {"n_cases": len(cases)}
    if not cases:
        return out
    # group by rollout length so we can batch
    by_len: dict[int, list] = {}
    for case in cases:
        e, t0, c, k, t_occ, t_re = case
        by_len.setdefault(t_re - t0 + extra, []).append(case)
    rows = {"probe": [], "explicit": []}
    for H, cs in by_len.items():
        items = [(e, t0, c) for (e, t0, c, *_rest) in cs]
        res = rollout_positions(exp, items, H, probe)
        for name, derr, pdisp in [("probe", res.disp_err, res.pred_disp_norm), ("explicit", res.explicit_disp_err, res.explicit_pred_disp_norm)]:
            if derr is None:
                continue
            for b, (e, t0, c, k, t_occ, t_re) in enumerate(cs):
                ep = store[e]
                gt_move = np.linalg.norm(ep.obj_pos[t_re, k, :2] - ep.obj_pos[t0, k, :2])
                rows[name].append({
                    "occluded_frames": t_re - t_occ,
                    "err_at_reappear": derr[b, t_re - t0 - 1, k],
                    "err_after": derr[b, t_re - t0 - 1 + extra, k] if t_re - t0 - 1 + extra < H else np.nan,
                    "err_during": derr[b, t_occ - t0 - 1 : t_re - t0 - 1, k].mean(),
                    "pred_move_during": pdisp[b, t_occ - t0 - 1 : t_re - t0 - 1, k].mean(),
                    "gt_move": gt_move,
                    "rollout_len": t_re - t0,
                })
    for name, rs in rows.items():
        if not rs:
            continue
        out[f"{name}/err_at_reappear_cm"] = _summ([r["err_at_reappear"] * 100 for r in rs])
        out[f"{name}/err_after_reappear_cm"] = _summ([r["err_after"] * 100 for r in rs])
        out[f"{name}/err_during_occlusion_cm"] = _summ([r["err_during"] * 100 for r in rs])
        out[f"{name}/pred_motion_during_occlusion_cm"] = _summ([r["pred_move_during"] * 100 for r in rs])
        out[f"{name}/gt_target_motion_cm"] = _summ([r["gt_move"] * 100 for r in rs])
        out[f"{name}/occluded_frames"] = _summ([r["occluded_frames"] for r in rs])
        out[f"{name}/rollout_len"] = _summ([r["rollout_len"] for r in rs])
    # control: visible static objects over rollouts of the typical length
    if control_episodes:
        lens = [r["rollout_len"] for r in rows["probe"]] or [4]
        H = int(round(np.median(lens)))
        ctrl_items = []
        ctrl_objs = []
        for e in control_episodes:
            ep = store[e]
            if ep.kind == "sweep":
                continue
            cam_list = cams if cams is not None else store.train_cams
            for c in cam_list:
                for j in range(store.n_objects):
                    static = np.linalg.norm(ep.obj_pos[H, j, :2] - ep.obj_pos[0, j, :2]) < 0.01
                    if static and _visible_throughout(ep, c, j, 0, H):
                        ctrl_items.append((e, 0, c))
                        ctrl_objs.append(j)
        ctrl_items, ctrl_objs = ctrl_items[:max_items], ctrl_objs[:max_items]
        if ctrl_items:
            res = rollout_positions(exp, ctrl_items, H, probe)
            for name, derr in [("probe", res.disp_err), ("explicit", res.explicit_disp_err)]:
                if derr is None:
                    continue
                errs = [derr[b, H - 1, j] for b, j in enumerate(ctrl_objs)]
                out[f"{name}/control_visible_static_err_cm"] = _summ(np.array(errs) * 100)
                out[f"{name}/control_len"] = H
    return out


# ------------------------------------------------------------- general error
def rollout_error_metrics(exp: Experiment, episodes: list[int], cams: list[int], horizon: int = 8, probe=None, ks=(1, 4, 8), max_items: int = 600) -> dict:
    """Displacement error over all objects / all episode kinds (overall prediction quality)."""
    store = exp.store
    items = [(e, 0, c) for e in episodes for c in cams][:max_items]
    if not items:
        return {}
    res = rollout_positions(exp, items, horizon, probe)
    out = {"n_rollouts": len(items)}
    for name, derr, aerr in [("probe", res.disp_err, res.abs_err), ("explicit", res.explicit_disp_err, None)]:
        if derr is None:
            continue
        for k in ks:
            if k <= horizon:
                out[f"{name}/k{k}/disp_err_cm"] = _summ(derr[:, k - 1] * 100)
                moved = res.gt_disp_norm[:, k - 1] > 0.02
                out[f"{name}/k{k}/disp_err_moved_cm"] = _summ(derr[:, k - 1][moved] * 100)
                if aerr is not None:
                    out[f"{name}/k{k}/abs_err_cm"] = _summ(aerr[:, k - 1] * 100)
    # readout floor: probe error on real tokens at t0
    out["probe/readout_err_cm"] = _summ(np.linalg.norm(res.ref_xy - res.gt_xy[:, 0], axis=-1) * 100)
    return out


@torch.no_grad()
def feature_error_metrics(exp: Experiment, episodes: list[int], cams: list[int], max_items: int = 300) -> dict:
    """1-step prediction error in token space (image-aligned representations only)."""
    if exp.is_gauss:
        return {}
    items = [(e, t, c) for e in episodes for t in range(exp.store.T) for c in cams]
    rng = np.random.default_rng(0)
    items = [items[i] for i in rng.permutation(len(items))[:max_items]]
    sampler = exp.sampler(1)
    tot, tot_copy = 0.0, 0.0
    for i in range(0, len(items), 32):
        b = sampler.batch(items[i : i + 32], 1)
        preds, _ = rollout_tokens(exp.predictor, exp.rep, b.tokens, b.actions)
        tot += ((preds[0].feat - b.targets[0].feat) ** 2).mean().item() * len(b.items)
        tot_copy += ((b.tokens.feat - b.targets[0].feat) ** 2).mean().item() * len(b.items)
    return {"feat_mse": tot / len(items), "feat_mse_copy_baseline": tot_copy / len(items)}


class NoChangePredictor(torch.nn.Module):
    """Baseline that predicts 'nothing moves' (zero residual / zero displacement).

    It scores perfectly on persistence and spurious motion by construction, which is
    exactly why it is needed: a representation's persistence number only means
    something next to its ability to move the objects that do move.
    """

    def __init__(self, predictor):
        super().__init__()
        self.cfg = predictor.cfg
        self.out_dim = predictor.cfg.out_dim

    def initial_memory(self, B):
        return None

    def forward(self, tokens, action, memory=None):
        from vrep.models.predictor import PredictorOutput

        return PredictorOutput(out=torch.zeros(tokens.B, tokens.N, self.out_dim, device=tokens.feat.device))


def evaluate_all(exp: Experiment, horizon: int = 8, max_items: int = 400, episodes: Optional[list[int]] = None, baseline: Optional[str] = None) -> dict:
    """Run every metric on validation episodes (or the given ones) for the training and held-out cameras.

    ``baseline="copy"`` replaces the predictor by the no-change baseline (same readouts).
    """
    store = exp.store
    if baseline == "copy":
        exp.predictor = NoChangePredictor(exp.predictor)
    elif baseline is not None:
        raise ValueError(baseline)
    val = exp.val_eps if episodes is None else list(episodes)
    results = {"rep": exp.rep.name, "run": str(exp.cfg.run), "data": str(store.root), "n_val_episodes": len(val), "baseline": baseline}
    views = {"train_cams": (store.train_cams, exp.probe)}
    if store.test_cams and exp.probe_heldout is not None:
        views["heldout_cam"] = (store.test_cams, exp.probe_heldout)
        views["heldout_cam_train_probe"] = (store.test_cams, exp.probe)
    if store.is_granular:
        from vrep.evaluation.granular import granular_metrics

        for vname, (cams, probe) in views.items():
            r = {"granular": granular_metrics(exp, val, cams, probe, max_items=max_items), "feature": feature_error_metrics(exp, val, cams)}
            results[vname] = r
            print(f"[{vname}] " + summarize_granular(r["granular"]))
        return results
    for vname, (cams, probe) in views.items():
        r = {}
        r["overall"] = rollout_error_metrics(exp, val, cams, horizon, probe, max_items=max_items)
        r["locality"] = locality_metrics(exp, val, cams, horizon, probe, max_items=max_items)
        r["persistence"] = persistence_metrics(exp, val, cams, probe, control_episodes=val, max_items=max_items)
        r["feature"] = feature_error_metrics(exp, val, cams)
        results[vname] = r
        print(f"[{vname}] " + summarize_view(r))
    return results


def summarize_granular(g: dict) -> str:
    def f(k):
        return g.get(k, {}).get("mean", float("nan"))

    H = g.get("horizon")
    return (f"change_err@4={f('probe/k4/change_err'):.3f} change_err@{H}={f(f'probe/k{H}/change_err'):.3f} "
            f"dug={f('probe/changed_err'):.3f} spurious={f('probe/spurious'):.3f} kept={f('probe/kept_ratio'):.2f} "
            f"grains/cell (n={g.get('n_rollouts')})")


def summarize_view(r: dict) -> str:
    def g(d, key, sub="mean"):
        v = d.get(key)
        return float("nan") if v is None else v[sub]

    o, l, p = r.get("overall", {}), r.get("locality", {}), r.get("persistence", {})
    parts = [
        f"readout_floor={g(o, 'probe/readout_err_cm'):.2f}cm",
        f"disp_err@1={g(o, 'probe/k1/disp_err_cm'):.2f}cm",
        f"disp_err@8={g(o, 'probe/k8/disp_err_cm'):.2f}cm",
        f"locality: target_err@4={g(l, 'probe/k4/target_disp_err_cm'):.2f}cm spurious@4={g(l, 'probe/k4/spurious_motion_cm'):.2f}cm",
        f"persistence: reappear={g(p, 'probe/err_at_reappear_cm'):.2f}cm control={g(p, 'probe/control_visible_static_err_cm'):.2f}cm (n={p.get('n_cases', 0)})",
    ]
    if "feat_mse" in r.get("feature", {}):
        parts.append(f"feat_mse={r['feature']['feat_mse']:.4f} (copy={r['feature']['feat_mse_copy_baseline']:.4f})")
    return " | ".join(parts)


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
