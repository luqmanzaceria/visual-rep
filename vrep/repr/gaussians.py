"""Dynamic Gaussians: an explicit 3D scene whose pieces the predictor moves.

A scene at time t is initialised from one camera's RGB-D frame: every pixel is
back-projected, points off the table are dropped, and a fixed number ``G`` of
points is kept. Each point becomes an isotropic Gaussian (centre, colour, scale,
opacity). The predictor receives each Gaussian as a token (features = colour,
scale, opacity; position = centre) and outputs a 3D displacement per Gaussian.

The *group* of a Gaussian (which object / pusher / table it was sampled from)
is stored for evaluation only; the model never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vrep.repr.base import Representation, TokenBatch
from vrep.repr.splat import render_gaussians
from vrep.sim.env import PUSHER_SEG_ID, backproject_depth

GROUP_TABLE = -1
GROUP_PUSHER = -2
GROUP_GRAIN = -3  # any grain of granular material (same id as the segmentation label)


@dataclass
class GaussianScene:
    mu: torch.Tensor  # (B,G,3)
    color: torch.Tensor  # (B,G,3)
    scale: torch.Tensor  # (B,G)
    opacity: torch.Tensor  # (B,G)
    group: torch.Tensor  # (B,G) int: object index, GROUP_PUSHER, GROUP_GRAIN or GROUP_TABLE
    valid: torch.Tensor  # (B,G) bool (padding)

    def to(self, device) -> "GaussianScene":
        return GaussianScene(*[x.to(device) for x in (self.mu, self.color, self.scale, self.opacity, self.group, self.valid)])

    def with_mu(self, mu: torch.Tensor) -> "GaussianScene":
        return GaussianScene(mu, self.color, self.scale, self.opacity, self.group, self.valid)

    @property
    def B(self) -> int:
        return self.mu.shape[0]

    @property
    def G(self) -> int:
        return self.mu.shape[1]

    def render(self, K, R, t, H, W, render_scale=1.0, background=None):
        return render_gaussians(self.mu, self.color, self.scale, self.opacity, K, R, t, H, W, render_scale, background, mask=self.valid)

    def centroids(self, n_objects: int) -> torch.Tensor:
        """(B, K, 3) centroid of the Gaussians of each object (NaN if none)."""
        out = []
        for k in range(n_objects):
            m = ((self.group == k) & self.valid).float()
            cnt = m.sum(1, keepdim=True)
            c = (self.mu * m[..., None]).sum(1) / cnt.clamp_min(1)
            c = torch.where(cnt > 0, c, torch.full_like(c, float("nan")))
            out.append(c)
        return torch.stack(out, 1)


def init_gaussians_from_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    seg: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    n_gaussians: int,
    rng: np.random.Generator,
    table_half: float,
    max_depth: float,
    opacity: float = 0.9,
    scale_mult: float = 1.0,
    object_boost: float = 3.0,
    z_min: float = -0.005,
) -> GaussianScene:
    """Build one Gaussian scene (batch of 1) from a single RGB-D(+seg) frame.

    ``seg`` is only used to (a) record groups for evaluation and (b) sample
    object pixels more densely (``object_boost``), which mimics a foreground
    aware point budget. The model never receives the groups.
    """
    H, W = depth.shape
    pts = backproject_depth(depth.astype(np.float32), K, R, t).reshape(-1, 3)
    cols = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    labels = seg.reshape(-1).astype(np.int64)
    d = depth.reshape(-1).astype(np.float32)
    keep = (d < max_depth * 0.999) & (pts[:, 2] > z_min) & (np.abs(pts[:, 0]) < table_half + 0.03) & (np.abs(pts[:, 1]) < table_half + 0.03)
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        raise ValueError("no valid points")
    # sampling weights: boost object / pusher pixels
    w = np.ones(idx.size, dtype=np.float64)
    fg = labels[idx] != -1
    w[fg] *= object_boost
    w /= w.sum()
    n_take = min(n_gaussians, idx.size)
    sel = rng.choice(idx, size=n_take, replace=False, p=w)
    # pixel footprint -> 3D scale: one pixel subtends depth/f metres; the selected
    # points are a subsample so enlarge by sqrt(points per sample)
    f = float(K[0, 0])
    density = idx.size / n_take
    scale = d[sel] / f * np.sqrt(density) * 0.6 * scale_mult
    # object points are denser than table points (boosted) -> smaller footprint
    scale = scale / np.sqrt(np.where(labels[sel] != -1, object_boost, 1.0))

    G = n_gaussians
    mu = np.zeros((G, 3), np.float32)
    col = np.zeros((G, 3), np.float32)
    sc = np.full((G,), 1e-3, np.float32)
    op = np.zeros((G,), np.float32)
    grp = np.full((G,), GROUP_TABLE, np.int64)
    valid = np.zeros((G,), bool)
    mu[:n_take] = pts[sel]
    col[:n_take] = cols[sel]
    sc[:n_take] = scale
    op[:n_take] = opacity
    lab = labels[sel]
    grp[:n_take] = np.where(lab == PUSHER_SEG_ID, GROUP_PUSHER, lab)
    valid[:n_take] = True
    return GaussianScene(
        torch.from_numpy(mu)[None],
        torch.from_numpy(col)[None],
        torch.from_numpy(sc)[None],
        torch.from_numpy(op)[None],
        torch.from_numpy(grp)[None],
        torch.from_numpy(valid)[None],
    )


def cat_scenes(scenes: list[GaussianScene]) -> GaussianScene:
    return GaussianScene(*[torch.cat([getattr(s, f) for s in scenes], 0) for f in ("mu", "color", "scale", "opacity", "group", "valid")])


def transform_gaussians_by_object_motion(
    scene: GaussianScene, obj_pos0: torch.Tensor, obj_quat0: torch.Tensor, obj_pos1: torch.Tensor, obj_quat1: torch.Tensor, pusher_delta: torch.Tensor
) -> torch.Tensor:
    """Ground-truth next centres (B,G,3) given rigid object motion and pusher translation.

    obj_pos* (B,K,3), obj_quat* (B,K,4) wxyz, pusher_delta (B,3). Table Gaussians stay.
    """
    mu = scene.mu.clone()
    B, K = obj_pos0.shape[:2]
    for k in range(K):
        m = scene.group == k
        if not m.any():
            continue
        R0 = quat_to_mat(obj_quat0[:, k])  # (B,3,3)
        R1 = quat_to_mat(obj_quat1[:, k])
        Rrel = R1 @ R0.transpose(1, 2)
        local = scene.mu - obj_pos0[:, k][:, None, :]
        moved = torch.einsum("bij,bgj->bgi", Rrel, local) + obj_pos1[:, k][:, None, :]
        mu = torch.where(m[..., None], moved, mu)
    mp = scene.group == GROUP_PUSHER
    mu = torch.where(mp[..., None], scene.mu + pusher_delta[:, None, :], mu)
    return mu


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(B,4) wxyz -> (B,3,3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


class GaussianRepresentation(Representation):
    """Tokens = [colour, scale, opacity] per Gaussian; positions = centres; output = displacement."""

    name = "gauss"
    pos_dim = 3
    residual = False
    feat_dim = 5
    out_dim = 3

    def __init__(self, cameras, table_half: float, max_depth: float, n_gaussians: int = 768, disp_scale: float = 0.05, **init_kwargs):
        self.cameras = cameras
        self.table_half = table_half
        self.max_depth = max_depth
        self.n_gaussians = n_gaussians
        self.disp_scale = disp_scale  # predictor output is multiplied by this (metres)
        self.init_kwargs = init_kwargs

    def scene_from_frame(self, rgb, depth, seg, cam: int, rng: np.random.Generator) -> GaussianScene:
        return init_gaussians_from_rgbd(
            rgb,
            depth.astype(np.float32),
            seg,
            self.cameras.K,
            self.cameras.R[cam],
            self.cameras.t[cam],
            self.n_gaussians,
            rng,
            self.table_half,
            self.max_depth,
            **self.init_kwargs,
        )

    def scene_from_store(self, store, items: list[tuple[int, int, int]], rng: np.random.Generator, device=None) -> GaussianScene:
        scenes = []
        for e, t, c in items:
            ep = store[e]
            scenes.append(self.scene_from_frame(ep.rgb[t, c], ep.depth[t, c], ep.seg[t, c], c, rng))
        s = cat_scenes(scenes)
        return s.to(device) if device is not None else s

    @staticmethod
    def tokens(scene: GaussianScene) -> TokenBatch:
        feat = torch.cat([scene.color, scene.scale[..., None] * 20.0, scene.opacity[..., None]], dim=-1)
        return TokenBatch(feat, scene.mu, scene.valid)

    def apply_prediction(self, tokens: TokenBatch, out: torch.Tensor) -> TokenBatch:
        return tokens.with_pos(tokens.pos + out * self.disp_scale)

    def target(self, next_tokens: TokenBatch, tokens: TokenBatch) -> torch.Tensor:
        return (next_tokens.pos - tokens.pos) / self.disp_scale
