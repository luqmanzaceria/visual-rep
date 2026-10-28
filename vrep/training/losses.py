"""Loss functions for the predictor."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vrep.repr.gaussians import GaussianScene


def feature_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """MSE between predicted and true token features (image-aligned representations)."""
    err = (pred - target) ** 2
    if mask is not None:
        err = err * mask[..., None]
        return err.sum() / (mask.sum() * pred.shape[-1]).clamp_min(1)
    return err.mean()


def render_loss(scene: GaussianScene, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, rgb_gt: torch.Tensor, depth_gt: torch.Tensor,
                fg: torch.Tensor, render_scale: float, w_rgb: float = 1.0, w_depth: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Photometric + depth loss of the rendered scene against an RGB-D frame.

    rgb_gt (B,h,w,3), depth_gt (B,h,w), fg (B,h,w) bool. The loss is restricted to
    foreground pixels (table + objects) because the Gaussians never cover the floor.
    Depth is compared as alpha-weighted depth so that partially covered pixels are
    not penalised for the missing background.
    """
    h, w = rgb_gt.shape[1:3]
    out = scene.render(K, R, t, h, w, render_scale)
    m = fg.float()
    n = m.sum().clamp_min(1)
    l_rgb = ((out["rgb"] - rgb_gt).abs().sum(-1) * m).sum() / n
    l_depth = ((out["depth"] - out["alpha"] * depth_gt).abs() * m).sum() / n
    loss = w_rgb * l_rgb + w_depth * l_depth
    return loss, {"rgb": l_rgb.detach(), "depth": l_depth.detach()}


def chamfer_loss(mu: torch.Tensor, valid: torch.Tensor, color: torch.Tensor, obs_pts: torch.Tensor, obs_col: torch.Tensor,
                 color_weight: float = 0.3, fwd_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric Chamfer distance between predicted Gaussian centres and observed points.

    mu (B,G,3), valid (B,G), color (B,G,3); obs_pts (B,P,3), obs_col (B,P,3).
    Colour is appended (scaled) to the coordinates so that a red Gaussian is
    matched to red pixels rather than to the nearby table.
    """
    a = torch.cat([mu, color * color_weight], -1)
    b = torch.cat([obs_pts, obs_col * color_weight], -1)
    d = torch.cdist(a, b)  # (B,G,P)
    big = torch.full_like(d, 1e3)
    d_masked = torch.where(valid[..., None], d, big)
    fm = valid if fwd_mask is None else (valid & fwd_mask)
    fwd = (d_masked.min(-1).values * fm).sum() / fm.sum().clamp_min(1)
    bwd = d_masked.min(1).values.mean()
    return fwd + bwd


@torch.no_grad()
def visible_mask(mu: torch.Tensor, valid: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, depth_obs: torch.Tensor,
                 render_scale: float, margin: float = 0.02) -> torch.Tensor:
    """(B,G) True for Gaussians that the camera would see in the observed frame.

    A Gaussian is visible if it projects inside the image and is not behind the
    observed surface at that pixel (z <= observed depth + margin). Hidden
    Gaussians receive no "move to the nearest observed point" gradient, so an
    object behind the arm is not dragged onto the arm's surface.
    """
    from vrep.repr.splat import project, world_to_camera

    Ks = K.clone()
    Ks[:2] *= render_scale
    u, v, z = project(world_to_camera(mu, R, t), Ks)
    h, w = depth_obs.shape[1:]
    ui, vi = u.round().long(), v.round().long()
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > 0)
    idx = (vi.clamp(0, h - 1) * w + ui.clamp(0, w - 1))
    d = depth_obs.reshape(depth_obs.shape[0], -1).gather(1, idx)
    return valid & inside & (z <= d + margin)


def oracle_loss(mu_pred: torch.Tensor, mu_gt: torch.Tensor, valid: torch.Tensor, disp_scale: float) -> torch.Tensor:
    err = ((mu_pred - mu_gt) / disp_scale) ** 2
    return (err.sum(-1) * valid).sum() / valid.sum().clamp_min(1)
