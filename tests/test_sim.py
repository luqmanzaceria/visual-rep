import numpy as np
import pytest

from vrep.sim import PushEnv, SceneConfig
from vrep.sim.env import PUSHER_SEG_ID, backproject_depth, project_points
from vrep.sim.scene import default_camera_split, look_at_xyaxes


@pytest.fixture(scope="module")
def env():
    return PushEnv(SceneConfig(image_size=64))


def test_look_at_axes_orthonormal():
    xy = look_at_xyaxes(np.array([1.0, 0.5, 0.8]), np.zeros(3))
    x, y = xy[:3], xy[3:]
    assert np.isclose(np.linalg.norm(x), 1) and np.isclose(np.linalg.norm(y), 1)
    assert np.isclose(x @ y, 0)
    # camera +y should point "up-ish"
    assert y[2] > 0


def test_camera_split():
    cfg = SceneConfig(n_cameras=4)
    train, test = default_camera_split(cfg)
    assert train == [0, 1, 2] and test == [3]
    az = cfg.azimuths_deg()
    assert len(az) == 4
    # held-out camera lies between two training cameras
    assert az[0] < az[3] < az[1]


def test_reset_places_objects_apart(env):
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = env.reset(rng)
        pos = s["obj_pos"]
        assert pos.shape == (env.n_objects, 3)
        assert np.all(np.abs(pos[:, :2]) < env.cfg.table_half)
        for i in range(env.n_objects):
            for j in range(i + 1, env.n_objects):
                assert np.linalg.norm(pos[i, :2] - pos[j, :2]) > 0.06
        # objects rest on the table
        assert np.all(pos[:, 2] > 0.0) and np.all(pos[:, 2] < 0.08)


def test_step_moves_pusher_by_action(env):
    rng = np.random.default_rng(1)
    env.reset(rng)
    env.teleport_pusher(np.array([0.0, 0.0]))
    p0 = env.pusher_position()
    env.step(np.array([0.04, -0.02]))
    p1 = env.pusher_position()
    assert np.allclose(p1[:2] - p0[:2], [0.04, -0.02], atol=1e-6)
    # clipping
    env.step(np.array([1.0, 0.0]))
    p2 = env.pusher_position()
    assert np.isclose(p2[0] - p1[0], env.cfg.max_action, atol=1e-6)


def test_pushing_moves_only_contacted_object(env):
    rng = np.random.default_rng(2)
    env.reset(rng)
    # put the red cube at the origin, other objects far away, pusher to the left
    env.teleport_object(0, np.array([0.0, 0.0]))
    env.teleport_object(1, np.array([0.2, 0.2]))
    env.teleport_object(2, np.array([-0.2, 0.2]))
    env.teleport_pusher(np.array([-0.08, 0.0]))
    before = env.object_positions().copy()
    for _ in range(4):
        env.step(np.array([0.05, 0.0]))
    after = env.object_positions()
    moved = np.linalg.norm(after[:, :2] - before[:, :2], axis=1)
    assert moved[0] > 0.05, f"target object did not move: {moved}"
    assert moved[1] < 5e-3 and moved[2] < 5e-3, f"untouched objects moved: {moved}"


def test_render_shapes_and_labels(env):
    rng = np.random.default_rng(3)
    env.reset(rng)
    out = env.render()
    C, H = env.cfg.n_cameras, env.cfg.image_size
    assert out["rgb"].shape == (C, H, H, 3) and out["rgb"].dtype == np.uint8
    assert out["depth"].shape == (C, H, H) and out["depth"].dtype == np.float32
    assert out["seg"].shape == (C, H, H) and out["seg"].dtype == np.int8
    labels = set(np.unique(out["seg"]).tolist())
    assert labels <= {-1, PUSHER_SEG_ID, 0, 1, 2}
    # every object and the pusher should be visible in at least one camera
    for k in range(env.n_objects):
        assert (out["seg"] == k).sum() > 10
    assert (out["seg"] == PUSHER_SEG_ID).sum() > 10
    assert np.all(out["depth"] > 0.3)


def test_backprojection_matches_object_positions(env):
    rng = np.random.default_rng(4)
    env.reset(rng)
    out = env.render()
    pos = env.object_positions()
    for c in range(env.cfg.n_cameras):
        pts = env.backproject(out["depth"][c], c)
        for k, spec in enumerate(env.cfg.objects):
            m = out["seg"][c] == k
            if m.sum() < 5:
                continue
            d = np.linalg.norm(pts[m] - pos[k], axis=1)
            # visible surface points lie within the object's extent of its centre
            assert np.median(d) < spec.radius * 1.3 + 0.01, (c, k, np.median(d))
            assert np.median(d) > 0.005


def test_project_backproject_roundtrip(env):
    rng = np.random.default_rng(5)
    env.reset(rng)
    out = env.render()
    c = 0
    pts = env.backproject(out["depth"][c], c)
    uv = env.project(pts, c)
    H = env.cfg.image_size
    v, u = np.mgrid[0:H, 0:H]
    assert np.allclose(uv[..., 0], u, atol=0.05)
    assert np.allclose(uv[..., 1], v, atol=0.05)


def test_project_objects_land_on_their_masks(env):
    rng = np.random.default_rng(6)
    env.reset(rng)
    out = env.render()
    pos = env.object_positions()
    for c in range(env.cfg.n_cameras):
        uv = env.project(pos, c)
        for k in range(env.n_objects):
            m = out["seg"][c] == k
            if m.sum() < 20:
                continue
            ys, xs = np.nonzero(m)
            # projected centre should lie close to the mask centroid
            assert abs(uv[k, 0] - xs.mean()) < 4 and abs(uv[k, 1] - ys.mean()) < 6, (c, k)


def test_state_roundtrip(env):
    rng = np.random.default_rng(7)
    env.reset(rng)
    s = env.get_state()
    img0 = env.render(depth=False, seg=False)["rgb"]
    for _ in range(3):
        env.step(rng.uniform(-0.05, 0.05, size=2))
    env.set_state(s)
    img1 = env.render(depth=False, seg=False)["rgb"]
    assert np.array_equal(img0, img1)


def test_pure_functions_consistent():
    K = np.array([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])
    R = np.eye(3)
    t = np.zeros(3)
    depth = np.full((64, 64), 2.0)
    pts = backproject_depth(depth, K, R, t)
    assert np.allclose(pts[..., 2], -2.0)
    uv = project_points(pts, K, R, t)
    v, u = np.mgrid[0:64, 0:64]
    assert np.allclose(uv[..., 0], u) and np.allclose(uv[..., 1], v)
