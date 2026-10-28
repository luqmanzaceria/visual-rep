"""Training loop for the predictor, followed by probe training."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from vrep.data.dataset import TransitionIndex
from vrep.training.builder import Experiment, build_experiment
from vrep.training.config import TrainConfig
from vrep.training.losses import chamfer_loss, feature_loss, oracle_loss, render_loss, visible_mask
from vrep.training.rollout import rollout_scene, rollout_tokens
from vrep.training.samplers import Batch2D, BatchGauss
from vrep.utils import count_parameters, seed_everything


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def horizon_at(step: int, cfg: TrainConfig) -> int:
    if cfg.horizon <= 1 or not cfg.horizon_schedule:
        return cfg.horizon
    p = min(1.0, step / max(1, cfg.steps // 2))
    return int(min(cfg.horizon, 1 + math.floor(p * cfg.horizon)))


def compute_loss(exp: Experiment, batch, horizon: int) -> tuple[torch.Tensor, dict]:
    cfg = exp.cfg
    logs = {}
    if isinstance(batch, Batch2D):
        preds, outs = rollout_tokens(exp.predictor, exp.rep, batch.tokens, batch.actions)
        losses = [feature_loss(p.feat, tgt.feat) for p, tgt in zip(preds, batch.targets)]
        loss = torch.stack(losses).mean()
        logs["feat"] = loss.detach()
        if exp.rep.name == "geo":
            logs["pm"] = torch.stack([((p.pos - tgt.pos) ** 2).mean() for p, tgt in zip(preds, batch.targets)]).mean().detach()
    else:
        assert isinstance(batch, BatchGauss)
        scenes, outs = rollout_scene(exp.predictor, exp.rep, batch.scene, batch.actions)
        K = exp.K
        rs = cfg.render_size / exp.store.image_size
        total = []
        for k, sc in enumerate(scenes):
            l = torch.zeros((), device=exp.device)
            if "render" in cfg.gauss_loss:
                lr, parts = render_loss(sc, K, batch.R, batch.t, batch.rgb[:, k], batch.depth[:, k], batch.fg[:, k], rs, cfg.w_rgb, cfg.w_depth)
                l = l + lr
                logs["rgb"] = logs.get("rgb", 0) + parts["rgb"] / len(scenes)
                logs["depth"] = logs.get("depth", 0) + parts["depth"] / len(scenes)
            if "chamfer" in cfg.gauss_loss and batch.obs_points is not None:
                fwd_mask = None
                if cfg.chamfer_visibility and batch.sup_depth is not None:
                    # visible in at least one supervision camera
                    B = sc.B
                    fwd_mask = torch.zeros_like(sc.valid)
                    for v in range(batch.sup_depth.shape[2]):
                        Rv = batch.sup_R[v][None].expand(B, -1, -1)
                        tv = batch.sup_t[v][None].expand(B, -1)
                        fwd_mask |= visible_mask(sc.mu, sc.valid, K, Rv, tv, batch.sup_depth[:, k, v], rs)
                    logs["visible"] = logs.get("visible", 0) + (fwd_mask.sum() / sc.valid.sum().clamp_min(1)).detach() / len(scenes)
                elif cfg.chamfer_visibility:
                    fwd_mask = visible_mask(sc.mu, sc.valid, K, batch.R, batch.t, batch.depth[:, k], rs)
                    logs["visible"] = logs.get("visible", 0) + (fwd_mask.sum() / sc.valid.sum().clamp_min(1)).detach() / len(scenes)
                lc = chamfer_loss(sc.mu, sc.valid, sc.color, batch.obs_points[:, k], batch.obs_colors[:, k], cfg.chamfer_color_weight, fwd_mask)
                l = l + cfg.w_chamfer * lc
                logs["chamfer"] = logs.get("chamfer", 0) + lc.detach() / len(scenes)
            if "oracle" in cfg.gauss_loss and batch.gt_mu is not None:
                lo = oracle_loss(sc.mu, batch.gt_mu[:, k], sc.valid, exp.rep.disp_scale)
                l = l + cfg.w_oracle * lo
                logs["oracle"] = logs.get("oracle", 0) + lo.detach() / len(scenes)
            total.append(l)
        loss = torch.stack(total).mean()
    if cfg.gate_l1 > 0 and outs[0].gate is not None:
        g = torch.stack([o.gate.mean() for o in outs]).mean()
        loss = loss + cfg.gate_l1 * g
        logs["gate"] = g.detach()
    logs["loss"] = loss.detach()
    return loss, logs


def _sample_items(index: TransitionIndex, rng: np.random.Generator, n: int, horizon: int, T: int):
    """Random (e, t, c) with t + horizon <= T."""
    out = []
    while len(out) < n:
        e, t, c = index[int(rng.integers(len(index)))]
        if t + horizon <= T:
            out.append((e, t, c))
    return out


@torch.no_grad()
def validate(exp: Experiment, sampler, index: TransitionIndex, rng: np.random.Generator, horizon: int, n_batches: int) -> dict:
    exp.predictor.eval()
    acc: dict[str, float] = {}
    for _ in range(n_batches):
        items = _sample_items(index, rng, exp.cfg.batch_size, horizon, exp.store.T)
        batch = sampler.batch(items, horizon)
        _, logs = compute_loss(exp, batch, horizon)
        for k, v in logs.items():
            acc[k] = acc.get(k, 0.0) + float(v) / n_batches
    exp.predictor.train()
    return acc


def train_grid_probe(exp: Experiment, probe, episodes: list[int], cams: list[int], steps: int, rng: np.random.Generator, log_prefix: str = "probe") -> dict:
    """Train a readout from ground-truth tokens to the grain occupancy map."""
    from vrep.evaluation.granular import OCC_NORM, grain_occupancy

    cfg = exp.cfg
    th = exp.store.scene.table_half
    index = TransitionIndex(episodes, cams, exp.store.T, history=1, horizon=0)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.probe_lr, weight_decay=0.01)
    probe.train()
    t0 = time.time()
    last = {}
    for step in range(steps):
        items = [index[int(i)] for i in rng.integers(len(index), size=cfg.probe_batch)]
        tokens = exp.encode_items(items, rng)
        occ = torch.from_numpy(np.stack([grain_occupancy(exp.store[e].grain_pos[t], th, probe.grid) for (e, t, c) in items])).to(exp.device)
        pred, _ = probe(tokens)
        loss = F.smooth_l1_loss(pred, occ, beta=0.1)
        for g in opt.param_groups:
            g["lr"] = cfg.probe_lr * (0.5 * (1 + math.cos(math.pi * step / steps)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            err = (pred - occ).abs().mean().item() * OCC_NORM
            last = {"step": step, "loss": loss.item(), "occ_err_grains": err}
            print(f"[{log_prefix}] step {step} loss {loss.item():.4f} occ_err {err:.3f} grains/cell ({time.time() - t0:.0f}s)")
    probe.eval()
    return last


def train_probe(exp: Experiment, probe, episodes: list[int], cams: list[int], steps: int, rng: np.random.Generator, log_prefix: str = "probe") -> dict:
    """Train a readout from ground-truth tokens to object (x, y) and visibility."""
    if exp.store.is_granular:
        return train_grid_probe(exp, probe, episodes, cams, steps, rng, log_prefix)
    cfg = exp.cfg
    index = TransitionIndex(episodes, cams, exp.store.T, history=1, horizon=0)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.probe_lr, weight_decay=0.01)
    probe.train()
    t0 = time.time()
    last = {}
    for step in range(steps):
        items = [index[int(i)] for i in rng.integers(len(index), size=cfg.probe_batch)]
        tokens = exp.encode_items(items, rng)
        xy = torch.from_numpy(np.stack([exp.store[e].obj_pos[t, :, :2] for (e, t, c) in items])).float().to(exp.device)
        vis = torch.from_numpy(np.stack([exp.store[e].visibility[t, c] > 0 for (e, t, c) in items])).float().to(exp.device)
        pxy, pvis = probe(tokens)
        l_xy = F.smooth_l1_loss(pxy, xy, beta=0.01)
        l_vis = F.binary_cross_entropy_with_logits(pvis, vis)
        loss = l_xy + 0.1 * l_vis
        for g in opt.param_groups:
            g["lr"] = cfg.probe_lr * (0.5 * (1 + math.cos(math.pi * step / steps)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            err = (pxy - xy).norm(dim=-1).mean().item()
            last = {"step": step, "loss": loss.item(), "xy_err_m": err, "vis": l_vis.item()}
            print(f"[{log_prefix}] step {step} loss {loss.item():.4f} xy_err {err * 100:.2f}cm vis {l_vis.item():.3f} ({time.time() - t0:.0f}s)")
    probe.eval()
    return last


def train_decoder(exp: Experiment, episodes: list[int], cams: list[int], steps: int, rng: np.random.Generator) -> dict:
    """Train the visualisation decoder (real tokens -> RGB) for image-aligned representations."""
    from vrep.training.samplers import _downsample_rgb

    cfg = exp.cfg
    dec = exp.decoder
    index = TransitionIndex(episodes, cams, exp.store.T, history=1, horizon=0)
    lr = 3e-4
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=0.01)
    dec.train()
    t0 = time.time()
    last = {}
    for step in range(steps):
        items = [index[int(i)] for i in rng.integers(len(index), size=cfg.probe_batch)]
        tokens = exp.encode_items(items, rng).to("cpu")
        target = torch.stack([_downsample_rgb(exp.store[e].rgb[t : t + 1, c], cfg.decoder_size)[0] for (e, t, c) in items])
        pred = dec(tokens)
        loss = F.mse_loss(pred, target)
        for g in opt.param_groups:
            g["lr"] = lr * (0.5 * (1 + math.cos(math.pi * step / steps)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            last = {"step": step, "mse": loss.item()}
            print(f"[decoder] step {step} mse {loss.item():.4f} ({time.time() - t0:.0f}s)")
    dec.eval()
    return last


def train(cfg: TrainConfig, store=None) -> Experiment:
    seed_everything(cfg.seed)
    exp = build_experiment(cfg, store)
    if exp.is_gauss and "oracle" in cfg.gauss_loss and exp.store.scene.n_grains > 0:
        raise ValueError("the oracle Gaussian loss tracks rigid objects only; use render/chamfer for granular scenes")
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"rep={cfg.rep} device={exp.device} episodes={len(exp.store)} train={len(exp.train_eps)} val={len(exp.val_eps)} "
          f"predictor params={count_parameters(exp.predictor) / 1e6:.2f}M")

    rng = np.random.default_rng(cfg.seed)
    T = exp.store.T
    train_index = TransitionIndex(exp.train_eps, exp.store.train_cams, T, history=1, horizon=1)
    val_index = TransitionIndex(exp.val_eps, exp.store.train_cams, T, history=1, horizon=1)
    heldout_index = TransitionIndex(exp.val_eps, exp.store.test_cams, T, history=1, horizon=1) if exp.store.test_cams else None
    sampler = exp.sampler(cfg.horizon, seed=cfg.seed)
    val_sampler = exp.sampler(cfg.horizon, seed=cfg.seed + 1)

    opt = torch.optim.AdamW(exp.predictor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    log_f = open(run_dir / "log.jsonl", "a")
    exp.predictor.train()
    t0 = time.time()
    running: dict[str, float] = {}
    for step in range(exp.step, cfg.steps):
        h = horizon_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        items = _sample_items(train_index, rng, cfg.batch_size, h, T)
        batch = sampler.batch(items, h)
        loss, logs = compute_loss(exp, batch, h)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(exp.predictor.parameters(), cfg.grad_clip)
        opt.step()
        for k, v in logs.items():
            running[k] = running.get(k, 0.0) + float(v)
        running["gn"] = running.get("gn", 0.0) + float(gn)
        exp.step = step + 1
        if (step + 1) % cfg.log_every == 0:
            avg = {k: v / cfg.log_every for k, v in running.items()}
            running = {}
            msg = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"step {step + 1}/{cfg.steps} h={h} lr={lr_at(step, cfg):.2e} {msg} ({time.time() - t0:.0f}s)")
            log_f.write(json.dumps({"step": step + 1, "horizon": h, **avg}) + "\n")
            log_f.flush()
        if (step + 1) % cfg.val_every == 0 or step + 1 == cfg.steps:
            v = validate(exp, val_sampler, val_index, rng, h, cfg.val_batches)
            rec = {"step": step + 1, "val": v}
            msg = f"  val: " + " ".join(f"{k}={x:.4f}" for k, x in v.items())
            if heldout_index is not None:
                vh = validate(exp, val_sampler, heldout_index, rng, h, cfg.val_batches)
                rec["val_heldout_cam"] = vh
                msg += " | held-out cam: " + " ".join(f"{k}={x:.4f}" for k, x in vh.items())
            print(msg)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            exp.save()
    log_f.close()

    # readouts: one on the training cameras, one on the held-out camera (evaluation only)
    probe_logs = {}
    if cfg.probe_steps > 0:
        probe_logs["probe"] = train_probe(exp, exp.probe, exp.train_eps, exp.store.train_cams, cfg.probe_steps, rng, "probe")
        if exp.probe_heldout is not None:
            probe_logs["probe_heldout"] = train_probe(exp, exp.probe_heldout, exp.train_eps, exp.store.test_cams, cfg.probe_steps, rng, "probe-heldout")
    if exp.decoder is not None and cfg.decoder_steps > 0:
        probe_logs["decoder"] = train_decoder(exp, exp.train_eps, exp.store.train_cams, cfg.decoder_steps, rng)
    exp.save()
    with open(run_dir / "probe.json", "w") as f:
        json.dump(probe_logs, f, indent=2)
    print("saved", run_dir / "checkpoint.pt")
    return exp
