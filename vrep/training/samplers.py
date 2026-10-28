"""Turn (episode, t, cam) triples into predictor training batches for each representation.

A batch holds the tokens at time t, the actions a_t..a_{t+H-1} and whatever the
loss needs for the H future frames: the true future tokens (image-aligned
representations) or the future RGB-D observations (Gaussians).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from vrep.data.dataset import EpisodeStore
from vrep.repr.base import TokenBatch
from vrep.repr.gaussians import GaussianRepresentation, GaussianScene, transform_gaussians_by_object_motion


@dataclass
class Batch2D:
    tokens: TokenBatch  # at t
    actions: torch.Tensor  # (B,H,2)
    targets: list[TokenBatch]  # true tokens at t+1..t+H
    items: list[tuple[int, int, int]]


@dataclass
class BatchGauss:
    scene: GaussianScene  # at t
    actions: torch.Tensor  # (B,H,2)
    rgb: torch.Tensor  # (B,H,h,w,3) float in [0,1] at render size
    depth: torch.Tensor  # (B,H,h,w) metres
    fg: torch.Tensor  # (B,H,h,w) bool: pixel lies on the table / objects (not floor)
    R: torch.Tensor  # (B,3,3)
    t: torch.Tensor  # (B,3)
    obs_points: Optional[torch.Tensor]  # (B,H,P,3) observed next-frame points (for chamfer)
    obs_colors: Optional[torch.Tensor]  # (B,H,P,3)
    gt_mu: Optional[torch.Tensor]  # (B,H,G,3) oracle future centres
    items: list[tuple[int, int, int]]
    # multi-view supervision (input is still one camera): observed depth of the next frames in every supervision camera
    sup_depth: Optional[torch.Tensor] = None  # (B,H,V,h,w)
    sup_R: Optional[torch.Tensor] = None  # (V,3,3)
    sup_t: Optional[torch.Tensor] = None  # (V,3)


def _downsample_rgb(rgb_u8: np.ndarray, size: int) -> torch.Tensor:
    x = torch.from_numpy(rgb_u8).float().div_(255).permute(0, 3, 1, 2)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="area")
    return x.permute(0, 2, 3, 1)


def _downsample_depth(depth: np.ndarray, size: int) -> torch.Tensor:
    x = torch.from_numpy(depth.astype(np.float32))[:, None]
    if x.shape[-1] != size:
        x = -F.max_pool2d(-x, kernel_size=x.shape[-1] // size)  # min-pool: keep nearest surface
    return x[:, 0]


class Sampler2D:
    def __init__(self, rep, store: EpisodeStore, horizon: int, device):
        self.rep = rep
        self.store = store
        self.horizon = horizon
        self.device = device

    def _tokens(self, items):
        if self.rep.name == "dino":
            return self.rep.tokens_from_cache(items, self.device)
        return self.rep.tokens_from_cache(self.store, items, self.device)

    def batch(self, items: list[tuple[int, int, int]], horizon: Optional[int] = None) -> Batch2D:
        H = horizon or self.horizon
        tokens = self._tokens(items)
        actions = torch.from_numpy(np.stack([self.store[e].actions[t : t + H] for (e, t, c) in items])).float()
        targets = [self._tokens([(e, t + k, c) for (e, t, c) in items]) for k in range(1, H + 1)]
        return Batch2D(tokens, actions.to(self.device), targets, items)


class SamplerGauss:
    def __init__(self, rep: GaussianRepresentation, store: EpisodeStore, horizon: int, device, render_size: int, seed: int = 0,
                 chamfer_points: int = 2048, need_oracle: bool = False, supervision_cams: Optional[list[int]] = None):
        self.rep = rep
        self.store = store
        self.horizon = horizon
        self.device = device
        self.render_size = render_size
        self.rng = np.random.default_rng(seed)
        self.chamfer_points = chamfer_points
        self.need_oracle = need_oracle
        self.cams = store.cameras
        self.supervision_cams = supervision_cams  # None: supervise with the input camera only

    def _observed_points(self, ep, t: int, c: int) -> tuple[np.ndarray, np.ndarray]:
        """Subsampled world points + colours at time t from camera c, or fused over all supervision cameras."""
        if self.supervision_cams:
            n = len(self.supervision_cams)
            parts = [self._observed_points_cam(ep, t, v, self.chamfer_points // n + (i < self.chamfer_points % n)) for i, v in enumerate(self.supervision_cams)]
            return np.concatenate([p for p, _ in parts]), np.concatenate([c_ for _, c_ in parts])
        return self._observed_points_cam(ep, t, c, self.chamfer_points)

    def _observed_points_cam(self, ep, t: int, c: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Subsampled world points + colours visible from camera c at time t (table region only)."""
        from vrep.sim.env import backproject_depth

        depth = ep.depth[t, c].astype(np.float32)
        pts = backproject_depth(depth, self.cams.K, self.cams.R[c], self.cams.t[c]).reshape(-1, 3)
        cols = ep.rgb[t, c].reshape(-1, 3).astype(np.float32) / 255.0
        th = self.rep.table_half
        keep = (depth.reshape(-1) < self.rep.max_depth * 0.999) & (pts[:, 2] > -0.005) & (np.abs(pts[:, 0]) < th + 0.03) & (np.abs(pts[:, 1]) < th + 0.03)
        idx = np.nonzero(keep)[0]
        sel = self.rng.choice(idx, size=n_points, replace=idx.size < n_points)
        return pts[sel].astype(np.float32), cols[sel]

    def batch(self, items: list[tuple[int, int, int]], horizon: Optional[int] = None) -> BatchGauss:
        H = horizon or self.horizon
        scene = self.rep.scene_from_store(self.store, items, self.rng)
        B = len(items)
        actions = torch.from_numpy(np.stack([self.store[e].actions[t : t + H] for (e, t, c) in items])).float()
        rgb = torch.stack([_downsample_rgb(self.store[e].rgb[t + 1 : t + H + 1, c], self.render_size) for (e, t, c) in items])
        depth = torch.stack([_downsample_depth(self.store[e].depth[t + 1 : t + H + 1, c], self.render_size) for (e, t, c) in items])
        # foreground = pixel on the table surface or above it (exclude floor / far background)
        fg = depth < self.rep.max_depth * 0.999
        R = torch.from_numpy(np.stack([self.cams.R[c] for (_, _, c) in items])).float()
        tt = torch.from_numpy(np.stack([self.cams.t[c] for (_, _, c) in items])).float()
        obs_p, obs_c = None, None
        if self.chamfer_points > 0:
            P, C = [], []
            for (e, t, c) in items:
                ps, cs = zip(*[self._observed_points(self.store[e], t + k, c) for k in range(1, H + 1)])
                P.append(np.stack(ps))
                C.append(np.stack(cs))
            obs_p = torch.from_numpy(np.stack(P))
            obs_c = torch.from_numpy(np.stack(C))
        gt_mu = None
        if self.need_oracle:
            mus = []
            for k in range(1, H + 1):
                p0 = torch.from_numpy(np.stack([self.store[e].obj_pos[t] for (e, t, c) in items]))
                q0 = torch.from_numpy(np.stack([self.store[e].obj_quat[t] for (e, t, c) in items]))
                p1 = torch.from_numpy(np.stack([self.store[e].obj_pos[t + k] for (e, t, c) in items]))
                q1 = torch.from_numpy(np.stack([self.store[e].obj_quat[t + k] for (e, t, c) in items]))
                pd = torch.from_numpy(np.stack([self.store[e].pusher_pos[t + k] - self.store[e].pusher_pos[t] for (e, t, c) in items]))
                mus.append(transform_gaussians_by_object_motion(scene, p0, q0, p1, q1, pd))
            gt_mu = torch.stack(mus, 1)
        d = self.device
        sup = {}
        if self.supervision_cams:
            V = self.supervision_cams
            sd = torch.stack([torch.stack([_downsample_depth(self.store[e].depth[t + 1 : t + H + 1, v], self.render_size) for v in V], 1) for (e, t, c) in items])
            sup = dict(
                sup_depth=sd.to(d),
                sup_R=torch.from_numpy(np.stack([self.cams.R[v] for v in V])).float().to(d),
                sup_t=torch.from_numpy(np.stack([self.cams.t[v] for v in V])).float().to(d),
            )
        return BatchGauss(
            scene.to(d), actions.to(d), rgb.to(d), depth.to(d), fg.to(d), R.to(d), tt.to(d),
            None if obs_p is None else obs_p.to(d), None if obs_c is None else obs_c.to(d),
            None if gt_mu is None else gt_mu.to(d), items, **sup,
        )
