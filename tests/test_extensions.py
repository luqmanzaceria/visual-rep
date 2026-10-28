"""Tests for the extensions: clutter / granular scenes, occupancy metrics, mono depth calibration, visibility-aware loss."""

import numpy as np
import pytest
import torch

from vrep.data import EpisodeStore, generate_dataset, generate_episode
from vrep.evaluation.granular import OCC_NORM, cell_centres, far_from_path, grain_occupancy
from vrep.repr.mono_depth import fit_affine_disparity, table_plane_depth
from vrep.sim import PushEnv
from vrep.sim.env import GRAIN_SEG_ID
from vrep.sim.scene import CLUTTER_OBJECTS, scene_preset
from vrep.training import TrainConfig, train
from vrep.training.losses import chamfer_loss, visible_mask


@pytest.fixture(scope="module")
def granular_env():
    return PushEnv(scene_preset("granular", image_size=48))


# ------------------------------------------------------------------ scenes
def test_presets():
    assert scene_preset("clutter").n_objects == 6
    g = scene_preset("granular", image_size=32)
    assert g.n_objects == 0 and g.n_grains > 0 and g.image_size == 32
    with pytest.raises(ValueError):
        scene_preset("nope")


def test_scene_roundtrip_through_dataset(tmp_path):
    cfg = scene_preset("clutter", image_size=40)
    root = generate_dataset(tmp_path / "c", n_episodes=2, T=2, cfg=cfg, workers=1)
    store = EpisodeStore(root)
    assert store.scene.objects == CLUTTER_OBJECTS and store.n_objects == 6
    assert store[0].obj_pos.shape == (3, 6, 3) and store[0].visibility.shape == (3, cfg.n_cameras, 6)
    assert not store.is_granular


def test_granular_pile_is_stable(granular_env):
    env = granular_env
    env.reset(np.random.default_rng(0))
    g0 = env.grain_positions().copy()
    assert g0.shape == (env.cfg.n_grains, 3)
    assert np.all(np.abs(g0[:, :2]) < env.cfg.table_half)
    far = -np.sign(g0[:, :2].mean(0)) * (env.cfg.table_half - 0.04)
    env.teleport_pusher(far)
    for _ in range(8):
        env.step(np.zeros(2))
    assert np.linalg.norm(env.grain_positions() - g0, axis=1).max() < 0.01


def test_plow_episode_changes_and_keeps_the_pile(granular_env):
    ep = generate_episode(granular_env, seed=1, T=16, kind="plow")
    assert ep.kind == "plow" and ep.grain_pos.shape == (17, granular_env.cfg.n_grains, 3)
    moved = np.linalg.norm(ep.grain_pos[-1, :, :2] - ep.grain_pos[0, :, :2], axis=1)
    assert (moved > 0.01).mean() > 0.3  # the plow changed the scene
    assert np.all(np.abs(ep.grain_pos[-1, :, :2]) < granular_env.cfg.table_half + 0.01)  # nothing spilled off
    assert GRAIN_SEG_ID in np.unique(ep.seg)
    assert ep.visibility.shape[-1] == 0


# --------------------------------------------------------------- occupancy
def test_grain_occupancy_counts():
    th = 0.28
    gp = np.array([[[-0.27, -0.27, 0], [-0.26, -0.26, 0], [0.27, 0.27, 0], [1.0, 1.0, 0]]], np.float32)
    occ = grain_occupancy(gp, th, grid=4)
    assert occ.shape == (1, 16)
    assert np.isclose(occ[0, 0] * OCC_NORM, 2) and np.isclose(occ[0, 15] * OCC_NORM, 1)
    assert np.isclose(occ.sum() * OCC_NORM, 3)  # the off-table grain is not counted


def test_far_from_path():
    th, grid = 0.28, 4
    cc = cell_centres(th, grid)
    path = np.array([[cc[0, 0], cc[0, 1]]])
    far = far_from_path(path, th, grid, dist=0.05)
    assert not far[0] and far[15]


# -------------------------------------------------------------- mono depth
def test_table_plane_depth_matches_render(push_episode, cameras, small_cfg):
    for c in range(cameras.n):
        plane, mask = table_plane_depth(cameras.K, cameras.R[c], cameras.t[c], 64, 64, small_cfg.table_half)
        true = push_episode.depth[0, c].astype(np.float32)
        table = mask & (push_episode.seg[0, c] == -1)
        assert table.sum() > 200
        assert np.median(np.abs(plane[table] - true[table])) < 0.005


def test_affine_fit_is_robust_to_objects():
    rng = np.random.default_rng(0)
    true_disp = rng.uniform(0.8, 1.6, size=(64, 64))
    rel = (true_disp - 0.3) / 2.5  # relative disparity: unknown scale and shift
    target = true_disp.copy()
    target[:20, :20] = 0.2  # a region that does not lie on the plane (an object)
    mask = np.ones_like(true_disp, dtype=bool)
    a, b = fit_affine_disparity(rel, target, mask)
    assert abs(a - 2.5) < 0.05 and abs(b - 0.3) < 0.05


# ------------------------------------------------------- visibility & loss
def test_visible_mask_and_masked_chamfer(cameras):
    cam = 0
    K = torch.from_numpy(cameras.K).float()
    R = torch.from_numpy(cameras.R[cam]).float()[None]
    t = torch.from_numpy(cameras.t[cam]).float()[None]
    cam_c = t[0]
    front = 0.5 * cam_c  # halfway to the camera
    behind = -0.2 * cam_c / cam_c.norm() + torch.tensor([0.0, 0.0, 0.0])
    mu = torch.stack([front, behind])[None]
    # observed depth: a wall at the "front" point's depth everywhere
    from vrep.repr.splat import project, world_to_camera

    _, _, z = project(world_to_camera(mu, R, t), K)
    depth = torch.full((1, 64, 64), float(z[0, 0]))
    vis = visible_mask(mu, torch.ones(1, 2, dtype=torch.bool), K, R, t, depth, render_scale=1.0)
    assert vis.tolist() == [[True, False]]
    obs = front[None, None].expand(1, 5, 3)
    col = torch.zeros(1, 5, 3)
    l_all = chamfer_loss(mu, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, 3), obs, col)
    l_vis = chamfer_loss(mu, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, 3), obs, col, fwd_mask=vis)
    assert l_vis < l_all  # the hidden Gaussian no longer pays for being far from the observed surface


# -------------------------------------------------------- end-to-end smoke
@pytest.mark.slow
@pytest.mark.parametrize("rep", ["dino", "gauss"])
def test_granular_training_and_metrics(tmp_path, rep):
    from vrep.evaluation.metrics import evaluate_all

    cfg = scene_preset("granular", image_size=56)
    root = generate_dataset(tmp_path / "g", n_episodes=4, T=4, cfg=cfg, workers=1, kind="plow")
    tc = TrainConfig(
        rep=rep, data=str(root), run=str(tmp_path / f"run_{rep}"), device="cpu", steps=2, batch_size=2, warmup=1, val_every=2,
        val_batches=1, log_every=1, probe_steps=2, probe_batch=2, d_model=32, n_layers=1, n_heads=2, probe_d_model=32, probe_layers=1,
        dino_input=56, n_gaussians=64, render_size=28, chamfer_points=64, frac_val=0.5, decoder_steps=0, chamfer_visibility=True,
    )
    exp = train(tc)
    res = evaluate_all(exp, horizon=4, max_items=4)
    g = res["train_cams"]["granular"]
    assert g["n_rollouts"] > 0 and np.isfinite(g["probe/k4/change_err"]["mean"])
    if rep == "gauss":
        assert "explicit/kept_ratio" in g


@pytest.mark.slow
def test_gauss_oracle_rejected_for_granular(tmp_path):
    cfg = scene_preset("granular", image_size=48)
    root = generate_dataset(tmp_path / "g", n_episodes=2, T=2, cfg=cfg, workers=1)
    tc = TrainConfig(rep="gauss", gauss_loss="oracle", data=str(root), run=str(tmp_path / "r"), device="cpu", steps=1)
    with pytest.raises(ValueError):
        train(tc)


@pytest.mark.slow
def test_multiview_supervision_batch_and_training(tmp_path):
    from vrep.training.builder import build_experiment

    cfg = scene_preset("default", image_size=48)
    root = generate_dataset(tmp_path / "d", n_episodes=3, T=3, cfg=cfg, workers=1)
    tc = TrainConfig(
        rep="gauss", data=str(root), run=str(tmp_path / "mv"), device="cpu", steps=2, batch_size=2, warmup=1, val_every=2, val_batches=1,
        log_every=1, probe_steps=1, probe_batch=2, d_model=32, n_layers=1, n_heads=2, probe_d_model=32, probe_layers=1, n_gaussians=64,
        render_size=24, chamfer_points=90, frac_val=0.34, chamfer_visibility=True, multiview_supervision=True,
    )
    exp = build_experiment(tc)
    b = exp.sampler(2).batch([(0, 0, 0), (1, 1, 2)], 2)
    assert b.sup_depth.shape == (2, 2, 3, 24, 24) and b.sup_R.shape == (3, 3, 3)
    assert b.obs_points.shape == (2, 2, 90, 3)  # fused over the three training cameras
    train(tc)
