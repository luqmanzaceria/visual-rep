"""Estimated (non-oracle) depth for the geometry-aware representation.

A monocular depth network (Depth Anything V2) predicts *relative* disparity,
i.e. disparity up to an unknown scale and shift. We recover metric depth with a
per-frame affine fit against the one thing a calibrated robot knows without
any per-frame ground truth: where the table plane is. Pixels whose ray hits the
table give a target disparity (1 / plane depth); the fit is made robust by
iteratively dropping the pixels that fit worst (objects and the pusher stick
out of the plane).

This turns the geometry representation from an upper bound (oracle depth)
into a realistic stand-in for a geometry-trained encoder such as DUSt3R/VGGT.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from vrep.repr.dino import IMAGENET_MEAN, IMAGENET_STD


def table_plane_depth(K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int, table_half: float, z: float = 0.0):
    """Depth along the optical axis of the plane z=``z`` for every pixel, and a mask of pixels that hit the table."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d_cam = np.stack([(u + 0.5 - cx) / fx, -(v + 0.5 - cy) / fy, -np.ones_like(u)], -1)  # z_c = 1 along the optical axis
    d_world = d_cam @ R.T
    s = (z - t[2]) / np.where(np.abs(d_world[..., 2]) < 1e-9, -1e-9, d_world[..., 2])  # ray parameter == optical-axis depth
    hit = t[None, None, :] + s[..., None] * d_world
    on_table = (s > 0) & (np.abs(hit[..., 0]) < table_half - 0.01) & (np.abs(hit[..., 1]) < table_half - 0.01)
    return s.astype(np.float32), on_table


def fit_affine_disparity(rel: np.ndarray, target_disp: np.ndarray, mask: np.ndarray, iters: int = 4, keep: float = 0.7) -> tuple[float, float]:
    """Robust least squares for ``target ≈ a * rel + b`` on ``mask`` pixels."""
    x = rel[mask].astype(np.float64)
    y = target_disp[mask].astype(np.float64)
    sel = np.ones_like(x, dtype=bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        A = np.stack([x[sel], np.ones(sel.sum())], 1)
        (a, b), *_ = np.linalg.lstsq(A, y[sel], rcond=None)
        r = np.abs(a * x + b - y)
        sel = r <= np.quantile(r, keep)
    return float(a), float(b)


class MonoDepth:
    def __init__(self, name: str = "depth-anything/Depth-Anything-V2-Small-hf", input_size: int = 252, device="cpu"):
        from transformers import AutoModelForDepthEstimation

        self.name = name
        self.input_size = input_size
        self.device = torch.device(device)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(self.device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def relative(self, rgb: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """uint8 (B,H,W,3) -> relative disparity (B,H,W) float32 at the input resolution."""
        B, H, W, _ = rgb.shape
        out = []
        for i in range(0, B, batch_size):
            x = torch.from_numpy(np.ascontiguousarray(rgb[i : i + batch_size])).to(self.device).float().div_(255).permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            d = self.model(pixel_values=x).predicted_depth  # (b, h, w)
            d = F.interpolate(d[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            out.append(d.float().cpu().numpy())
        return np.concatenate(out, 0)

    def metric(self, rgb: np.ndarray, cams: list[int], cameras, table_half: float, max_depth: float, rel: Optional[np.ndarray] = None) -> np.ndarray:
        """Metric depth (B,H,W) for frames ``rgb`` taken by cameras ``cams``, calibrated on the table plane."""
        rel = self.relative(rgb) if rel is None else rel
        B, H, W = rel.shape
        out = np.empty_like(rel)
        for i, c in enumerate(cams):
            plane, mask = table_plane_depth(cameras.K, cameras.R[c], cameras.t[c], H, W, table_half)
            a, b = fit_affine_disparity(rel[i], 1.0 / np.maximum(plane, 1e-3), mask)
            disp = a * rel[i] + b
            out[i] = np.clip(1.0 / np.maximum(disp, 1.0 / max_depth), 0.05, max_depth)
        return out
