import numpy as np
import pytest
import torch

from vrep.repr.gaussians import (
    GROUP_PUSHER,
    GROUP_TABLE,
    cat_scenes,
    init_gaussians_from_rgbd,
    transform_gaussians_by_object_motion,
)
from vrep.repr.splat import project, render_gaussians, world_to_camera
from vrep.sim.env import project_points


def _cam_tensors(cameras, cam, B=1):
    K = torch.from_numpy(cameras.K).float()
    R = torch.from_numpy(cameras.R[cam]).float()[None].expand(B, -1, -1)
    t = torch.from_numpy(cameras.t[cam]).float()[None].expand(B, -1)
    return K, R, t


def test_projection_matches_numpy(cameras):
    pts = np.array([[0.1, -0.05, 0.03], [-0.2, 0.1, 0.0], [0.0, 0.0, 0.1]])
    for cam in range(cameras.n):
        K, R, t = _cam_tensors(cameras, cam)
        mu = torch.from_numpy(pts).float()[None]
        u, v, z = project(world_to_camera(mu, R, t), K)
        uv_np = project_points(pts, cameras.K, cameras.R[cam], cameras.t[cam])
        assert np.allclose(u[0].numpy(), uv_np[:, 0], atol=1e-3)
        assert np.allclose(v[0].numpy(), uv_np[:, 1], atol=1e-3)
        assert torch.all(z > 0)


def test_single_gaussian_peaks_at_projection(cameras):
    cam = 1
    K, R, t = _cam_tensors(cameras, cam)
    H = W = 64
    mu = torch.tensor([[[0.05, -0.1, 0.04]]])
    out = render_gaussians(mu, torch.ones(1, 1, 3), torch.tensor([[0.02]]), torch.tensor([[0.9]]), K, R, t, H, W, render_scale=1.0)
    img = out["rgb"][0, ..., 0]
    peak = torch.nonzero(img == img.max())[0]
    uv = project_points(mu[0].numpy(), cameras.K, cameras.R[cam], cameras.t[cam])[0]
    assert abs(peak[1].item() - uv[0]) <= 1.0 and abs(peak[0].item() - uv[1]) <= 1.0
    assert out["alpha"].max() > 0.5
    # depth at the peak equals the Gaussian's camera depth
    _, _, z = project(world_to_camera(mu, R, t), K)
    d = out["depth"][0, peak[0], peak[1]] / out["alpha"][0, peak[0], peak[1]]
    assert abs(d.item() - z.item()) < 1e-3


def test_front_to_back_occlusion(cameras):
    cam = 0
    K, R, t = _cam_tensors(cameras, cam)
    cam_centre = torch.from_numpy(cameras.t[cam]).float()
    far = torch.tensor([0.0, 0.0, 0.0])
    near = far + 0.3 * (cam_centre - far)  # same ray, closer to the camera
    mu = torch.stack([far, near])[None]  # far listed first on purpose
    color = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    out = render_gaussians(mu, color, torch.tensor([[0.03, 0.03]]), torch.tensor([[0.95, 0.95]]), K, R, t, 64, 64)
    uv = project_points(far.numpy(), cameras.K, cameras.R[cam], cameras.t[cam])
    px = out["rgb"][0, int(round(uv[1])), int(round(uv[0]))]
    assert px[2] > 0.8 and px[0] < 0.1, px  # blue (near) wins


def test_render_is_differentiable_wrt_positions(cameras):
    K, R, t = _cam_tensors(cameras, 2)
    mu = torch.tensor([[[0.0, 0.0, 0.03], [0.1, 0.0, 0.03]]], requires_grad=True)
    out = render_gaussians(mu, torch.rand(1, 2, 3), torch.tensor([[0.02, 0.02]]), torch.tensor([[0.9, 0.9]]), K, R, t, 32, 32, render_scale=0.5)
    out["rgb"].sum().backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all() and mu.grad.abs().sum() > 0


def test_init_from_rgbd_reproduces_image(push_episode, cameras, small_cfg):
    rng = np.random.default_rng(0)
    ep = push_episode
    cam = 0
    scene = init_gaussians_from_rgbd(
        ep.rgb[0, cam], ep.depth[0, cam].astype(np.float32), ep.seg[0, cam], cameras.K, cameras.R[cam], cameras.t[cam],
        n_gaussians=1500, rng=rng, table_half=small_cfg.table_half, max_depth=small_cfg.max_depth,
    )
    assert scene.mu.shape == (1, 1500, 3) and scene.valid.sum() > 1000
    groups = set(scene.group[scene.valid].unique().tolist())
    assert GROUP_TABLE in groups and GROUP_PUSHER in groups and len(groups & {0, 1, 2}) >= 2
    K, R, t = _cam_tensors(cameras, cam)
    out = scene.render(K, R, t, 64, 64)
    img = out["rgb"][0].numpy()
    target = ep.rgb[0, cam].astype(np.float32) / 255.0
    # compare where the Gaussians cover the image (the table region)
    m = out["alpha"][0].numpy() > 0.5
    assert m.mean() > 0.3
    err = np.abs(img[m] - target[m]).mean()
    assert err < 0.15, err
    # object centroids are close to the true object centres (within the object radius)
    cent = scene.centroids(3)[0].numpy()
    for k, spec in enumerate(small_cfg.objects):
        if np.isfinite(cent[k]).all():
            assert np.linalg.norm(cent[k] - ep.obj_pos[0, k]) < spec.radius * 1.2


def test_transform_by_object_motion(push_episode, cameras, small_cfg):
    rng = np.random.default_rng(1)
    ep = push_episode
    cam = 1
    scene = init_gaussians_from_rgbd(
        ep.rgb[0, cam], ep.depth[0, cam].astype(np.float32), ep.seg[0, cam], cameras.K, cameras.R[cam], cameras.t[cam],
        n_gaussians=800, rng=rng, table_half=small_cfg.table_half, max_depth=small_cfg.max_depth,
    )
    T = ep.T
    p0, q0 = torch.from_numpy(ep.obj_pos[0])[None], torch.from_numpy(ep.obj_quat[0])[None]
    p1, q1 = torch.from_numpy(ep.obj_pos[T])[None], torch.from_numpy(ep.obj_quat[T])[None]
    pd = torch.from_numpy(ep.pusher_pos[T] - ep.pusher_pos[0])[None]
    mu1 = transform_gaussians_by_object_motion(scene, p0, q0, p1, q1, pd)
    delta = (mu1 - scene.mu)[0]
    k = ep.target
    moved_gt = np.linalg.norm(ep.obj_pos[T, k, :2] - ep.obj_pos[0, k, :2])
    for g in range(3):
        m = scene.group[0] == g
        if m.sum() == 0:
            continue
        d = delta[m].norm(dim=-1).mean().item()
        gt = np.linalg.norm(ep.obj_pos[T, g, :2] - ep.obj_pos[0, g, :2])
        if g == k and moved_gt > 0.01:
            assert abs(d - gt) < 0.35 * gt + 0.01, (d, gt)
        else:
            assert d < 0.01
    assert delta[scene.group[0] == GROUP_TABLE].abs().max() == 0
    pusher_delta = delta[scene.group[0] == GROUP_PUSHER]
    assert pusher_delta.shape[0] > 0
    assert torch.allclose(pusher_delta, pd[0].float().expand_as(pusher_delta), atol=1e-6)


def test_cat_scenes(push_episode, cameras, small_cfg):
    rng = np.random.default_rng(2)
    ep = push_episode
    s = [
        init_gaussians_from_rgbd(ep.rgb[0, c], ep.depth[0, c].astype(np.float32), ep.seg[0, c], cameras.K, cameras.R[c], cameras.t[c],
                                 n_gaussians=100, rng=rng, table_half=small_cfg.table_half, max_depth=small_cfg.max_depth)
        for c in range(2)
    ]
    sc = cat_scenes(s)
    assert sc.B == 2 and sc.G == 100
