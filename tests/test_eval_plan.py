"""Evaluation metrics and planning on a tiny trained run (CPU)."""

import numpy as np
import pytest
import torch

from vrep.data import generate_dataset
from vrep.evaluation.metrics import (
    evaluate_all,
    find_occlusion,
    locality_metrics,
    persistence_metrics,
    rollout_error_metrics,
    rollout_positions,
)
from vrep.planning import ModelCost, OracleCost, RandomCost, cem_plan, make_tasks, run_mpc
from vrep.sim import PushEnv
from vrep.sim.scene import SceneConfig
from vrep.training import TrainConfig, train


@pytest.fixture(scope="module")
def tiny_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("data") / "tiny"
    cfg = SceneConfig(image_size=56)
    # a few episodes of each kind so every metric has something to chew on
    generate_dataset(root, n_episodes=4, T=8, cfg=cfg, workers=1, kind="push")
    generate_dataset(root, n_episodes=4, T=8, seed0=100, cfg=cfg, workers=1, kind="sweep")
    runs = {}
    for rep in ["dino", "gauss"]:
        tc = TrainConfig(
            rep=rep, data=str(root), run=str(tmp_path_factory.mktemp("runs") / rep), device="cpu", steps=2, batch_size=2, warmup=1,
            val_every=2, val_batches=1, log_every=1, probe_steps=2, probe_batch=2, d_model=32, n_layers=1, n_heads=2,
            probe_d_model=32, probe_layers=1, dino_input=56, n_gaussians=48, render_size=28, chamfer_points=64, frac_val=0.5,
        )
        runs[rep] = train(tc)
    return runs


def test_find_occlusion_synthetic():
    class Ep:
        visibility = np.zeros((10, 1, 1), dtype=np.int32)

    ep = Ep()
    ep.visibility[:, 0, 0] = [100, 100, 90, 10, 0, 5, 80, 100, 100, 100]
    assert find_occlusion(ep, 0, 0) == (2, 3, 6)
    # gradual occlusion: the rollout starts at the last fully visible frame
    ep.visibility[:, 0, 0] = [100, 100, 60, 40, 10, 0, 50, 80, 100, 100]
    assert find_occlusion(ep, 0, 0) == (1, 4, 7)
    ep.visibility[:, 0, 0] = 100
    assert find_occlusion(ep, 0, 0) is None
    ep.visibility[:, 0, 0] = [0, 0, 50, 100, 100, 100, 100, 100, 100, 100]
    assert find_occlusion(ep, 0, 0) is None  # occluded from the start
    ep.visibility[:, 0, 0] = [100, 100, 0, 0, 0, 0, 0, 0, 0, 0]
    assert find_occlusion(ep, 0, 0) is None  # never reappears does not count


@pytest.mark.slow
@pytest.mark.parametrize("rep", ["dino", "gauss"])
def test_rollout_positions_shapes(tiny_runs, rep):
    exp = tiny_runs[rep]
    items = [(e, 0, c) for e in exp.val_eps[:2] for c in [0, 1]]
    res = rollout_positions(exp, items, horizon=3)
    B, K = len(items), exp.store.n_objects
    assert res.pred_xy.shape == (B, 3, K, 2) and res.ref_xy.shape == (B, K, 2) and res.gt_xy.shape == (B, 4, K, 2)
    assert res.disp_err.shape == (B, 3, K) and np.isfinite(res.disp_err).all()
    if rep == "gauss":
        assert res.explicit_xy is not None and res.explicit_xy.shape == (B, 4, K, 2)
        # a barely trained predictor (2 steps from a zero-init head) moves almost nothing
        assert np.nanmax(res.explicit_pred_disp_norm) < 0.01


@pytest.mark.slow
def test_metrics_run(tiny_runs):
    exp = tiny_runs["gauss"]
    eps = list(range(len(exp.store)))
    loc = locality_metrics(exp, eps, [0], horizon=4, ks=(1, 4))
    assert "probe/k1/spurious_motion_cm" in loc and loc["n_rollouts"] > 0
    per = persistence_metrics(exp, eps, None, control_episodes=eps)
    assert "n_cases" in per
    ov = rollout_error_metrics(exp, eps, [0, 1], horizon=4, ks=(1, 4))
    assert "probe/readout_err_cm" in ov
    res = evaluate_all(exp, horizon=4, max_items=8)
    assert "train_cams" in res and "heldout_cam" in res


def test_cem_finds_minimum():
    rng = np.random.default_rng(0)
    target = np.array([[0.03, -0.02], [-0.01, 0.04]])
    seq, cost = cem_plan(lambda A: ((A - target) ** 2).sum((1, 2)), horizon=2, action_dim=2, bound=0.06, rng=rng, n_samples=128, n_iters=6)
    assert np.allclose(seq, target, atol=0.01) and cost < 1e-3


@pytest.mark.slow
def test_mpc_with_model_and_baselines(tiny_runs):
    exp = tiny_runs["dino"]
    env = PushEnv(exp.store.scene)
    tasks = make_tasks(env, 1)
    for cost in [ModelCost(exp, cam=0), RandomCost()]:
        r = run_mpc(env, tasks[0], cost, horizon=2, max_steps=2, n_samples=4, n_elite=2, n_iters=1)
        assert set(r) >= {"success", "final_dist", "init_dist", "steps"} and r["steps"] <= 2
    exp_g = tiny_runs["gauss"]
    r = run_mpc(env, tasks[0], ModelCost(exp_g, cam=1, readout="explicit"), horizon=2, max_steps=1, n_samples=4, n_elite=2, n_iters=1)
    assert r["steps"] == 1


def test_oracle_mpc_reaches_goal():
    cfg = SceneConfig(image_size=32)
    env = PushEnv(cfg, render=False)
    tasks = make_tasks(env, 2, dist=(0.08, 0.10))
    cost = OracleCost(cfg)
    results = [run_mpc(env, t, cost, horizon=4, max_steps=10, n_samples=32, n_elite=6, n_iters=2, seed=i) for i, t in enumerate(tasks)]
    assert all(r["final_dist"] < r["init_dist"] for r in results)
    assert any(r["success"] for r in results)
