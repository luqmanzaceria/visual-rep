"""Geometry-aware 2D features (a stand-in for DUSt3R / VGGT style encoders).

DUSt3R/VGGT-type models output, for every patch, both a feature vector and a 3D
*pointmap* (where that patch lies in space). We emulate this family with
DINOv2 patch features augmented by a per-patch pointmap. By default the pointmap
comes from the simulator's depth (an *oracle* pointmap: the upper bound of what
a geometry-trained encoder could provide). ``pointmap_source="estimated"`` plugs
in an external estimator instead (hook left for VGGT / monocular depth).

The tokens stay image-aligned (one per patch of one camera) so the predictor's
job is the same as for DINOv2: predict the next frame's tokens. The difference is
that every token now *knows where it is in the world*, which is what the
hypothesis says should help viewpoint consistency and locality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from vrep.repr.base import Representation, TokenBatch
from vrep.repr.dino import DinoEncoder, DinoRepresentation, FeatureCache
from vrep.sim.env import backproject_depth


def patch_pointmap(depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, grid: int, max_depth: float) -> np.ndarray:
    """Per-patch mean world point (grid*grid, 3) from a depth map (H, W).

    Background pixels (depth >= max_depth) are excluded from the average; a patch
    that is entirely background gets the far point along its central ray.
    """
    pts = backproject_depth(depth.astype(np.float32), K, R, t)  # (H,W,3)
    H, W = depth.shape
    valid = (depth < max_depth * 0.999).astype(np.float32)
    ph, pw = H / grid, W / grid
    out = np.zeros((grid * grid, 3), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            ys, ye = int(round(i * ph)), int(round((i + 1) * ph))
            xs, xe = int(round(j * pw)), int(round((j + 1) * pw))
            p = pts[ys:ye, xs:xe].reshape(-1, 3)
            v = valid[ys:ye, xs:xe].reshape(-1)
            if v.sum() > 0:
                out[i * grid + j] = (p * v[:, None]).sum(0) / v.sum()
            else:
                out[i * grid + j] = p.mean(0)
    return out


def patch_pointmap_torch(depth: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, grid: int, max_depth: float) -> torch.Tensor:
    """Batched version: depth (B,H,W) -> (B, grid*grid, 3). R (B,3,3), t (B,3)."""
    B, H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth.float()
    x = (u.float() + 0.5 - cx) / fx * z
    y = -(v.float() + 0.5 - cy) / fy * z
    pc = torch.stack([x, y, -z], dim=-1)  # (B,H,W,3)
    pw = torch.einsum("bhwj,bij->bhwi", pc, R) + t[:, None, None, :]
    valid = (z < max_depth * 0.999).float()
    # average-pool points and validity down to the patch grid
    pw_p = F.adaptive_avg_pool2d((pw * valid[..., None]).permute(0, 3, 1, 2), grid)  # (B,3,g,g)
    v_p = F.adaptive_avg_pool2d(valid[:, None], grid)  # (B,1,g,g)
    all_p = F.adaptive_avg_pool2d(pw.permute(0, 3, 1, 2), grid)
    out = torch.where(v_p > 0, pw_p / v_p.clamp_min(1e-6), all_p)
    return out.permute(0, 2, 3, 1).reshape(B, grid * grid, 3)


class PointmapCache:
    """Per-patch oracle pointmaps for a whole dataset: (E, T+1, C, grid*grid, 3) float32."""

    def __init__(self, path):
        self.path = Path(path)
        self.arr = np.load(self.path, mmap_mode="r")

    def get(self, e: int, t: int, c: int) -> np.ndarray:
        return np.asarray(self.arr[e, t, c])

    @staticmethod
    def build(store, grid: int, max_depth: float, path) -> "PointmapCache":
        path = Path(path)
        E, T1, C = len(store), store.T + 1, store.cameras.n
        shape = (E, T1, C, grid * grid, 3)
        if path.exists() and np.load(path, mmap_mode="r").shape == shape:
            return PointmapCache(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        K = torch.from_numpy(store.cameras.K).float()
        R = torch.from_numpy(store.cameras.R).float()
        t = torch.from_numpy(store.cameras.t).float()
        for e in range(E):
            depth = torch.from_numpy(store[e].depth.astype(np.float32))  # (T+1, C, H, W)
            d = depth.reshape(-1, *depth.shape[2:])
            Rb = R[None].expand(T1, -1, -1, -1).reshape(-1, 3, 3)
            tb = t[None].expand(T1, -1, -1).reshape(-1, 3)
            pm = patch_pointmap_torch(d, K, Rb, tb, grid, max_depth)
            arr[e] = pm.reshape(T1, C, grid * grid, 3).numpy()
        arr.flush()
        del arr
        return PointmapCache(path)


def build_mono_pointmap_cache(store, grid: int, path, device="cpu", log_every: int = 50) -> PointmapCache:
    """Point maps from *estimated* depth (Depth Anything V2 + table-plane calibration).

    Also writes ``<path>.json`` with the depth error against the simulator, so the
    quality of the estimate is reported alongside results.
    """
    import json
    import time

    from vrep.repr.mono_depth import MonoDepth

    path = Path(path)
    E, T1, C = len(store), store.T + 1, store.cameras.n
    shape = (E, T1, C, grid * grid, 3)
    info_path = path.with_suffix(".json")
    if path.exists() and info_path.exists() and np.load(path, mmap_mode="r").shape == shape:
        return PointmapCache(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = MonoDepth(device=device)
    scene = store.scene
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    K = torch.from_numpy(store.cameras.K).float()
    R = torch.from_numpy(store.cameras.R).float()
    t = torch.from_numpy(store.cameras.t).float()
    err_table, err_obj = [], []
    t0 = time.time()
    for e in range(E):
        ep = store[e]
        rgb = ep.rgb.reshape(-1, *ep.rgb.shape[2:])
        cams = [c for _ in range(T1) for c in range(C)]
        depth = md.metric(rgb, cams, store.cameras, scene.table_half, scene.max_depth)
        true = ep.depth.reshape(-1, *ep.depth.shape[2:]).astype(np.float32)
        seg = ep.seg.reshape(-1, *ep.seg.shape[2:])
        fg = seg != -1
        bg = (seg == -1) & (true < scene.max_depth * 0.999)
        err_obj.append(float(np.median(np.abs(depth - true)[fg])) if fg.any() else np.nan)
        err_table.append(float(np.median(np.abs(depth - true)[bg])))
        Rb = R[None].expand(T1, -1, -1, -1).reshape(-1, 3, 3)
        tb = t[None].expand(T1, -1, -1).reshape(-1, 3)
        pm = patch_pointmap_torch(torch.from_numpy(depth), K, Rb, tb, grid, scene.max_depth)
        arr[e] = pm.reshape(T1, C, grid * grid, 3).numpy()
        if e % log_every == 0:
            print(f"[mono pointmap] {e}/{E} depth err table {np.nanmean(err_table) * 100:.2f}cm objects {np.nanmean(err_obj) * 100:.2f}cm ({time.time() - t0:.0f}s)")
    arr.flush()
    del arr
    with open(info_path, "w") as f:
        json.dump({"estimator": md.name, "median_depth_err_table_m": float(np.nanmean(err_table)),
                   "median_depth_err_objects_m": float(np.nanmean(err_obj)), "shape": list(shape)}, f, indent=2)
    return PointmapCache(path)


class GeometryRepresentation(Representation):
    """Tokens = [DINO features, world pointmap]; positions = world pointmap (3D)."""

    name = "geo"
    pos_dim = 3
    residual = True

    def __init__(
        self,
        dino: DinoRepresentation,
        cameras,
        max_depth: float,
        point_scale: float = 1.0,
        pointmap_source: str = "oracle",
        pointmap_cache: Optional[PointmapCache] = None,
    ):
        self.dino = dino
        self.cameras = cameras
        self.max_depth = max_depth
        self.point_scale = point_scale  # multiply world coordinates so they have a similar range to features
        self.pointmap_source = pointmap_source
        self.feat_dim = dino.feat_dim + 3
        self.out_dim = self.feat_dim
        self.grid = dino.encoder.grid
        self.pointmap_cache = pointmap_cache
        self._pm_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def pointmap(self, store, e: int, t: int, c: int) -> np.ndarray:
        if self.pointmap_cache is not None:
            return self.pointmap_cache.get(e, t, c)
        key = (e, t, c)
        pm = self._pm_cache.get(key)
        if pm is None:
            if self.pointmap_source != "oracle":
                raise NotImplementedError("only the oracle pointmap is implemented for cached datasets")
            depth = store[e].depth[t, c].astype(np.float32)
            pm = patch_pointmap(depth, self.cameras.K, self.cameras.R[c], self.cameras.t[c], self.grid, self.max_depth)
            self._pm_cache[key] = pm
        return pm

    def tokens_from_cache(self, store, items: list[tuple[int, int, int]], device=None) -> TokenBatch:
        base = self.dino.tokens_from_cache(items)
        pm = torch.from_numpy(np.stack([self.pointmap(store, *it) for it in items]).astype(np.float32)) * self.point_scale
        feat = torch.cat([base.feat, pm], dim=-1)
        tb = TokenBatch(feat, pm.clone())
        return tb.to(device) if device is not None else tb

    @torch.no_grad()
    def tokens_from_rgbd(self, rgb: np.ndarray, depth: np.ndarray, cams: list[int], device=None) -> TokenBatch:
        """``depth`` is ignored (replaced by an estimate) when the representation uses mono point maps."""
        if self.pointmap_source == "mono":
            if getattr(self, "_mono", None) is None:
                from vrep.repr.mono_depth import MonoDepth

                self._mono = MonoDepth(device="cuda" if torch.cuda.is_available() else "cpu")  # MPS lacks bicubic upsampling
            depth = self._mono.metric(rgb, cams, self.cameras, self.table_half, self.max_depth)
        base = self.dino.tokens_from_rgb(rgb)
        pm = np.stack(
            [
                patch_pointmap(depth[i].astype(np.float32), self.cameras.K, self.cameras.R[c], self.cameras.t[c], self.grid, self.max_depth)
                for i, c in enumerate(cams)
            ]
        )
        pm = torch.from_numpy(pm) * self.point_scale
        tb = TokenBatch(torch.cat([base.feat, pm], dim=-1), pm.clone())
        return tb.to(device) if device is not None else tb

    def apply_prediction(self, tokens: TokenBatch, out: torch.Tensor) -> TokenBatch:
        feat = tokens.feat + out
        return TokenBatch(feat, feat[..., -3:], tokens.mask)

    def target(self, next_tokens: TokenBatch, tokens: TokenBatch) -> torch.Tensor:
        return next_tokens.feat - tokens.feat
