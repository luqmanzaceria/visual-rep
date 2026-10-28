import numpy as np
import pytest

from vrep.data import Episode, EpisodeStore, TransitionIndex, generate_dataset, generate_episode
from vrep.sim import PushEnv, SceneConfig
from vrep.sim.policies import EPISODE_KINDS, make_plan, plan_push, plan_sweep


@pytest.fixture(scope="module")
def env():
    return PushEnv(SceneConfig(image_size=64))


@pytest.mark.parametrize("kind", EPISODE_KINDS)
def test_plans_produce_valid_actions(env, kind):
    rng = np.random.default_rng(10)
    env.reset(rng)
    plan = make_plan(env, rng, kind)
    assert plan.kind == kind
    env.teleport_pusher(plan.start_xy)
    for t in range(12):
        a = plan.policy(env, t)
        assert a.shape == (2,) and np.all(np.isfinite(a))
        env.step(a)


def test_push_plan_moves_target(env):
    rng = np.random.default_rng(11)
    moved = []
    for _ in range(8):
        env.reset(rng)
        plan = plan_push(env, rng)
        env.teleport_pusher(plan.start_xy)
        before = env.object_positions()[plan.target, :2].copy()
        for t in range(8):
            env.step(plan.policy(env, t))
        moved.append(np.linalg.norm(env.object_positions()[plan.target, :2] - before))
    # most pushes should displace the target by a few centimetres
    assert np.median(moved) > 0.03, moved


def test_sweep_plan_occludes_target(env):
    """A sweep should hide the target from the chosen camera for at least one frame."""
    rng = np.random.default_rng(12)
    hits = 0
    n = 10
    for _ in range(n):
        env.reset(rng)
        plan = plan_sweep(env, rng)
        env.teleport_pusher(plan.start_xy)
        vis = []
        for t in range(16):
            seg = env.render(depth=False)["seg"][plan.cam]
            vis.append((seg == plan.target).sum())
            env.step(plan.policy(env, t))
        vis = np.array(vis)
        full = vis.max()
        if full > 0 and vis.min() < 0.25 * full:
            hits += 1
    assert hits >= n // 2, f"only {hits}/{n} sweeps produced an occlusion"


def test_generate_episode_shapes(env):
    ep = generate_episode(env, seed=3, T=5)
    C, H, K = env.cfg.n_cameras, env.cfg.image_size, env.n_objects
    assert ep.rgb.shape == (6, C, H, H, 3)
    assert ep.depth.shape == (6, C, H, H) and ep.depth.dtype == np.float16
    assert ep.seg.shape == (6, C, H, H)
    assert ep.obj_pos.shape == (6, K, 3) and ep.obj_quat.shape == (6, K, 4)
    assert ep.pusher_pos.shape == (6, 3)
    assert ep.actions.shape == (5, 2) and np.all(np.abs(ep.actions) <= env.cfg.max_action + 1e-6)
    assert ep.visibility.shape == (6, C, K)
    assert ep.kind in EPISODE_KINDS
    # pusher motion is consistent with the recorded actions (unless clipped at the edge)
    d = ep.pusher_pos[1:, :2] - ep.pusher_pos[:-1, :2]
    inside = np.all(np.abs(ep.pusher_pos[1:, :2]) < env.cfg.table_half - env.cfg.pusher_radius - 1e-4, axis=1)
    assert np.allclose(d[inside], ep.actions[inside], atol=1e-5)


def test_episode_is_deterministic(env):
    a = generate_episode(env, seed=5, T=4)
    b = generate_episode(env, seed=5, T=4)
    assert np.array_equal(a.rgb, b.rgb) and np.allclose(a.obj_pos, b.obj_pos) and a.kind == b.kind


def test_dataset_roundtrip(tmp_path):
    cfg = SceneConfig(image_size=48)
    out = generate_dataset(tmp_path / "ds", n_episodes=4, T=3, cfg=cfg, workers=1)
    store = EpisodeStore(out)
    assert len(store) == 4 and store.T == 3
    assert store.image_size == 48 and store.cameras.n == cfg.n_cameras
    ep = store[0]
    assert isinstance(ep, Episode) and ep.rgb.shape[0] == 4
    # camera parameters stored in the dataset agree with a fresh environment
    env = PushEnv(cfg, render=False)
    R, t = env.all_extrinsics()
    assert np.allclose(store.cameras.R, R) and np.allclose(store.cameras.t, t)
    assert np.allclose(store.cameras.K, env.K)
    train, val = store.split(frac_val=0.25)
    assert len(val) == 1 and len(train) == 3 and not set(train) & set(val)
    assert store.train_cams == [0, 1, 2] and store.test_cams == [3]


def test_transition_index():
    idx = TransitionIndex(episodes=[0, 2], cams=[0, 1], T=5, history=2, horizon=2)
    # t ranges over [history-1, T-horizon] = [1, 3] -> 3 values per episode per cam
    assert len(idx) == 2 * 3 * 2
    assert idx[0] == (0, 1, 0)
    ts = {t for (_, t, _) in idx.items}
    assert ts == {1, 2, 3}
