"""A small, fully differentiable Gaussian splatting renderer in pure PyTorch.

It is dense (every Gaussian contributes to every pixel) which is fine for the
scale we use (<= ~1k Gaussians, <= 64x64 pixels) and keeps the gradient exact.
Gaussians are isotropic in 3D; their image footprint is a 2D isotropic Gaussian
with radius ``f * scale / depth``. Compositing is front-to-back alpha blending.
"""

from __future__ import annotations

import torch


def world_to_camera(mu: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """mu (B,G,3) world -> camera frame (B,G,3). R (B,3,3) cam->world, t (B,3).

    p_cam = R^T (p_world - t), i.e. the same convention as ``vrep.sim.env.project_points``.
    """
    return torch.einsum("bji,bgj->bgi", R, mu - t[:, None, :])


def project(mu_cam: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Camera-frame points (B,G,3) -> pixel u, v (B,G) and positive depth z (B,G)."""
    z = -mu_cam[..., 2]
    zc = z.clamp_min(1e-4)
    u = K[0, 0] * mu_cam[..., 0] / zc + K[0, 2] - 0.5
    v = -K[1, 1] * mu_cam[..., 1] / zc + K[1, 2] - 0.5
    return u, v, z


def render_gaussians(
    mu: torch.Tensor,
    color: torch.Tensor,
    scale: torch.Tensor,
    opacity: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    H: int,
    W: int,
    render_scale: float = 1.0,
    background: torch.Tensor | None = None,
    max_alpha: float = 0.99,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Render a batch of Gaussian scenes.

    mu (B,G,3) world centres; color (B,G,3) in [0,1]; scale (B,G) metres (std);
    opacity (B,G) in [0,1]; K (3,3) intrinsics for an image of ``H x W`` pixels
    (K is scaled by ``render_scale`` when rendering at a different resolution);
    R (B,3,3), t (B,3) camera-to-world extrinsics.

    Returns rgb (B,H,W,3), depth (B,H,W) (alpha-weighted mean depth), alpha (B,H,W).
    """
    B, G, _ = mu.shape
    dev = mu.device
    Ks = K.clone()
    Ks[:2] *= render_scale
    mu_cam = world_to_camera(mu, R, t)
    u, v, z = project(mu_cam, Ks)
    in_front = z > 1e-3
    if mask is not None:
        in_front = in_front & mask
    sigma = (Ks[0, 0] * scale / z.clamp_min(1e-3)).clamp(0.3, 64.0)  # pixels

    # sort front to back
    order = torch.argsort(torch.where(in_front, z, torch.full_like(z, 1e6)), dim=1)
    u, v, z = u.gather(1, order), v.gather(1, order), z.gather(1, order)
    sigma, in_front = sigma.gather(1, order), in_front.gather(1, order)
    color = color.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
    opacity = opacity.gather(1, order)

    py, px = torch.meshgrid(torch.arange(H, device=dev, dtype=mu.dtype), torch.arange(W, device=dev, dtype=mu.dtype), indexing="ij")
    px = px.reshape(1, 1, H * W)
    py = py.reshape(1, 1, H * W)
    d2 = (u[..., None] - px) ** 2 + (v[..., None] - py) ** 2  # (B,G,HW)
    g = torch.exp(-0.5 * d2 / (sigma[..., None] ** 2))
    alpha = (opacity[..., None] * g).clamp(max=max_alpha) * in_front[..., None]
    log_T = torch.cumsum(torch.log1p(-alpha), dim=1)  # inclusive
    T = torch.exp(log_T - torch.log1p(-alpha))  # exclusive transmittance
    w = T * alpha  # (B,G,HW)
    rgb = torch.einsum("bgp,bgc->bpc", w, color)
    depth = torch.einsum("bgp,bg->bp", w, z)
    acc = w.sum(1)
    if background is not None:
        rgb = rgb + (1 - acc)[..., None] * background.view(1, 1, 3)
    return {
        "rgb": rgb.view(B, H, W, 3),
        "depth": depth.view(B, H, W),
        "alpha": acc.view(B, H, W),
    }
