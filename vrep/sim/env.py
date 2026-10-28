"""MuJoCo pushing environment with multi-camera RGB / depth / segmentation rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from vrep.sim.scene import SceneConfig, build_mjcf, intrinsics

PUSHER_SEG_ID = -2  # segmentation id for the pusher (objects are 0..K-1, background -1)
GRAIN_SEG_ID = -3  # segmentation id shared by all grains of granular material


@dataclass
class EnvState:
    """Full simulator state needed to restore a scene exactly."""

    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray

    def copy(self) -> "EnvState":
        return EnvState(self.qpos.copy(), self.qvel.copy(), self.mocap_pos.copy(), self.mocap_quat.copy())


class PushEnv:
    """A tabletop with a few objects and a kinematic pusher.

    Actions are 2D pusher displacements in the table plane (world frame), in
    metres, clipped to ``[-max_action, max_action]``. One ``step`` moves the
    pusher linearly over ``substeps`` physics steps.
    """

    def __init__(self, cfg: SceneConfig | None = None, render: bool = True):
        self.cfg = cfg or SceneConfig()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.cfg))
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        if render:
            self._renderer = mujoco.Renderer(self.model, self.cfg.image_size, self.cfg.image_size)

        self.pusher_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
        self.pusher_mocap = self.model.body_mocapid[self.pusher_body]
        self.obj_bodies = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec.name) for spec in self.cfg.objects
        ]
        self.obj_qpos_addr = [self.model.jnt_qposadr[self.model.body_jntadr[b]] for b in self.obj_bodies]
        self.obj_qvel_addr = [self.model.jnt_dofadr[self.model.body_jntadr[b]] for b in self.obj_bodies]
        self.cam_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, f"cam{i}") for i in range(self.cfg.n_cameras)
        ]
        # geom id -> segmentation label
        self._geom_label = np.full(self.model.ngeom, -1, dtype=np.int8)
        for k, b in enumerate(self.obj_bodies):
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == b:
                    self._geom_label[g] = k
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == self.pusher_body:
                self._geom_label[g] = PUSHER_SEG_ID
        self.grain_bodies = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"grain{i}") for i in range(self.cfg.n_grains)]
        self.grain_qpos_addr = [self.model.jnt_qposadr[self.model.body_jntadr[b]] for b in self.grain_bodies]
        grain_set = set(self.grain_bodies)
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] in grain_set:
                self._geom_label[g] = GRAIN_SEG_ID
        self.K = intrinsics(self.cfg)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ basics
    @property
    def n_objects(self) -> int:
        return self.cfg.n_objects

    @property
    def image_size(self) -> int:
        return self.cfg.image_size

    def get_state(self) -> EnvState:
        d = self.data
        return EnvState(d.qpos.copy(), d.qvel.copy(), d.mocap_pos.copy(), d.mocap_quat.copy())

    def set_state(self, s: EnvState) -> None:
        d = self.data
        d.qpos[:] = s.qpos
        d.qvel[:] = s.qvel
        d.mocap_pos[:] = s.mocap_pos
        d.mocap_quat[:] = s.mocap_quat
        mujoco.mj_forward(self.model, d)

    def object_positions(self) -> np.ndarray:
        """(K, 3) object centre positions in world frame."""
        return self.data.xpos[self.obj_bodies].astype(np.float32).reshape(-1, 3)

    def object_quats(self) -> np.ndarray:
        return self.data.xquat[self.obj_bodies].astype(np.float32).reshape(-1, 4)

    def grain_positions(self) -> np.ndarray:
        """(G, 3) grain centres (empty if the scene has no granular material)."""
        return self.data.xpos[self.grain_bodies].astype(np.float32).reshape(-1, 3)

    def pusher_position(self) -> np.ndarray:
        return self.data.mocap_pos[self.pusher_mocap].astype(np.float32).copy()

    def observation_state(self) -> dict[str, np.ndarray]:
        """Compact ground-truth state (what the metrics are computed against)."""
        return {
            "obj_pos": self.object_positions(),
            "obj_quat": self.object_quats(),
            "pusher_pos": self.pusher_position(),
            "grain_pos": self.grain_positions(),
        }

    # ------------------------------------------------------------------- reset
    def reset(self, rng: np.random.Generator, min_gap: float = 0.02, margin: float = 0.05) -> dict[str, np.ndarray]:
        """Place objects and pusher at random non-overlapping positions on the table."""
        cfg = self.cfg
        lim = cfg.table_half - margin
        placed: list[tuple[np.ndarray, float]] = []

        def sample(radius: float) -> np.ndarray:
            for _ in range(500):
                p = rng.uniform(-lim, lim, size=2)
                if all(np.linalg.norm(p - q) > radius + r + min_gap for q, r in placed):
                    placed.append((p, radius))
                    return p
            raise RuntimeError("could not place object")

        mujoco.mj_resetData(self.model, self.data)
        for k, spec in enumerate(cfg.objects):
            p = sample(spec.radius)
            adr = self.obj_qpos_addr[k]
            yaw = rng.uniform(0, 2 * np.pi) if spec.geom_type == "box" else 0.0
            quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            self.data.qpos[adr : adr + 3] = [p[0], p[1], spec.half_height + 1e-3]
            self.data.qpos[adr + 3 : adr + 7] = quat
            self.data.qvel[self.obj_qvel_addr[k] : self.obj_qvel_addr[k] + 6] = 0.0
        if cfg.n_grains > 0:
            # a pile: grains on a jittered grid, stacked in layers, inside a square footprint
            # keep the pile well inside the table so plowing through it does not spill grains over the walls
            c_lim = cfg.table_half - cfg.pile_half - 0.07
            centre = rng.uniform(-c_lim, c_lim, size=2)
            placed.append((centre, cfg.pile_half * 1.45))
            r = cfg.grain_radius
            n_side = max(1, int(np.floor(2 * cfg.pile_half / (2.05 * r))))
            slots = [(i, j, l) for l in range(64) for i in range(n_side) for j in range(n_side)]
            sp = 2.05 * r
            for gi, (i, j, l) in enumerate(slots[: cfg.n_grains]):
                # odd layers sit in the pockets of the layer below (stable, like a packed mound)
                off = (np.array([i, j]) - (n_side - 1) / 2) * sp + (l % 2) * sp / 2 + rng.uniform(-0.05, 0.05, size=2) * r
                adr = self.grain_qpos_addr[gi]
                self.data.qpos[adr : adr + 3] = [centre[0] + off[0], centre[1] + off[1], r + l * np.sqrt(2) / 2 * sp + 1e-3]
                self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        pp = sample(cfg.pusher_radius)
        self.data.mocap_pos[self.pusher_mocap] = [pp[0], pp[1], cfg.pusher_z]
        self.data.mocap_quat[self.pusher_mocap] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        # let objects (and the pile) settle on the table
        for _ in range(20 if cfg.n_grains == 0 else 1500):
            mujoco.mj_step(self.model, self.data)
        if cfg.n_grains > 0:
            self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)  # sync derived quantities (xpos) with qpos
        return self.observation_state()

    def teleport_pusher(self, xy: np.ndarray) -> None:
        """Move the pusher instantly (used by scripted policies between pushes)."""
        xy = np.clip(np.asarray(xy, dtype=float), -self.cfg.table_half + 0.02, self.cfg.table_half - 0.02)
        self.data.mocap_pos[self.pusher_mocap, :2] = xy
        mujoco.mj_forward(self.model, self.data)

    def teleport_object(self, k: int, xy: np.ndarray) -> None:
        adr = self.obj_qpos_addr[k]
        self.data.qpos[adr : adr + 2] = np.asarray(xy, dtype=float)
        self.data.qvel[self.obj_qvel_addr[k] : self.obj_qvel_addr[k] + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=float)[:2], -cfg.max_action, cfg.max_action)
        start = self.data.mocap_pos[self.pusher_mocap, :2].copy()
        lim = cfg.table_half - cfg.pusher_radius
        end = np.clip(start + a, -lim, lim)
        for i in range(1, cfg.substeps + 1):
            t = i / cfg.substeps
            self.data.mocap_pos[self.pusher_mocap, :2] = start + t * (end - start)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # sync derived quantities (xpos) with qpos
        return self.observation_state()

    # ------------------------------------------------------------------ render
    def render(self, cam_ids: Sequence[int] | None = None, depth: bool = True, seg: bool = True):
        """Render all requested cameras.

        Returns a dict with ``rgb`` (C,H,W,3) uint8, ``depth`` (C,H,W) float32 metres
        (distance along the optical axis) and ``seg`` (C,H,W) int8 labels
        (``-1`` background/table/walls, ``0..K-1`` objects, ``-2`` pusher).
        """
        assert self._renderer is not None, "env created with render=False"
        r = self._renderer
        cam_ids = list(range(self.cfg.n_cameras)) if cam_ids is None else list(cam_ids)
        rgbs, depths, segs = [], [], []
        for c in cam_ids:
            cam = self.cam_ids[c]
            r.disable_depth_rendering()
            r.disable_segmentation_rendering()
            r.update_scene(self.data, camera=cam)
            rgbs.append(r.render().copy())
            if depth:
                r.enable_depth_rendering()
                r.update_scene(self.data, camera=cam)
                dm = np.minimum(r.render().astype(np.float32), self.cfg.max_depth)
                depths.append(dm)
                r.disable_depth_rendering()
            if seg:
                r.enable_segmentation_rendering()
                r.update_scene(self.data, camera=cam)
                s = r.render()[..., 0]
                lab = np.full(s.shape, -1, dtype=np.int8)
                valid = s >= 0
                lab[valid] = self._geom_label[s[valid]]
                segs.append(lab)
                r.disable_segmentation_rendering()
        out = {"rgb": np.stack(rgbs)}
        if depth:
            out["depth"] = np.stack(depths)
        if seg:
            out["seg"] = np.stack(segs)
        return out

    # ------------------------------------------------------------- camera math
    def camera_extrinsics(self, cam: int) -> tuple[np.ndarray, np.ndarray]:
        """Rotation (3,3) and translation (3,) mapping camera coords -> world coords."""
        cid = self.cam_ids[cam]
        R = self.data.cam_xmat[cid].reshape(3, 3).copy()
        t = self.data.cam_xpos[cid].copy()
        return R, t

    def all_extrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        Rs, ts = zip(*[self.camera_extrinsics(c) for c in range(self.cfg.n_cameras)])
        return np.stack(Rs).astype(np.float32), np.stack(ts).astype(np.float32)

    def backproject(self, depth: np.ndarray, cam: int, K: np.ndarray | None = None) -> np.ndarray:
        """Back-project a depth map (H,W) to world-frame points (H,W,3)."""
        return backproject_depth(depth, K if K is not None else self.K, *self.camera_extrinsics(cam))

    def project(self, points: np.ndarray, cam: int) -> np.ndarray:
        """Project world points (...,3) to pixel coordinates (...,2) (u right, v down)."""
        R, t = self.camera_extrinsics(cam)
        return project_points(points, self.K, R, t)


def backproject_depth(depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Depth (H,W) along the optical axis -> world points (H,W,3).

    MuJoCo cameras look along -z with +y up, while pixel v grows downward.
    """
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth.astype(np.float64)
    x = (u + 0.5 - cx) / fx * z
    y = -(v + 0.5 - cy) / fy * z
    pc = np.stack([x, y, -z], axis=-1)
    return pc @ R.T + t


def project_points(points: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World points (...,3) -> pixel coords (...,2) using the same convention as ``backproject_depth``."""
    pc = (np.asarray(points, dtype=np.float64) - t) @ R
    z = -pc[..., 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K[0, 0] * pc[..., 0] / z + K[0, 2] - 0.5
    v = -K[1, 1] * pc[..., 1] / z + K[1, 2] - 0.5
    return np.stack([u, v], axis=-1)
