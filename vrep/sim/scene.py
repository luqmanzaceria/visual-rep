"""Tabletop pushing scene description (MJCF generation + camera geometry).

The scene is deliberately simple so that the *representation* is the only thing
that changes between experiments:

* a flat table with low walls,
* three objects with fixed identity (shape + colour) placed at random,
* a tall cylindrical "pusher" that stands in for a robot arm. It is tall so
  that, seen from an elevated camera, it hides whatever is behind it. That is
  what the persistence experiment relies on,
* a ring of cameras looking at the table centre. Some are used for training,
  one is held out for the viewpoint-consistency experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    geom_type: str  # box | sphere | cylinder
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]

    @property
    def radius(self) -> float:
        """Footprint radius in the table plane (used for placement)."""
        if self.geom_type == "box":
            return float(np.hypot(self.size[0], self.size[1]))
        return float(self.size[0])

    @property
    def half_height(self) -> float:
        if self.geom_type == "box":
            return float(self.size[2])
        if self.geom_type == "sphere":
            return float(self.size[0])
        return float(self.size[1])


# Fixed object identities. Every episode contains all of them, which makes
# "where is the red cube" a well-defined readout target across episodes.
OBJECTS: tuple[ObjectSpec, ...] = (
    ObjectSpec("red_cube", "box", (0.04, 0.04, 0.04), (0.90, 0.15, 0.15, 1.0)),
    ObjectSpec("green_sphere", "sphere", (0.04,), (0.15, 0.75, 0.20, 1.0)),
    ObjectSpec("blue_cylinder", "cylinder", (0.04, 0.04), (0.20, 0.30, 0.90, 1.0)),
)

# Extra identities for the clutter experiment (locality with more bystanders).
EXTRA_OBJECTS: tuple[ObjectSpec, ...] = (
    ObjectSpec("yellow_cube", "box", (0.035, 0.035, 0.035), (0.92, 0.80, 0.15, 1.0)),
    ObjectSpec("purple_sphere", "sphere", (0.035,), (0.60, 0.25, 0.75, 1.0)),
    ObjectSpec("orange_cylinder", "cylinder", (0.035, 0.035), (0.95, 0.50, 0.10, 1.0)),
)
CLUTTER_OBJECTS = OBJECTS + EXTRA_OBJECTS


def objects_to_dicts(objs) -> list[dict]:
    return [{"name": o.name, "geom_type": o.geom_type, "size": list(o.size), "rgba": list(o.rgba)} for o in objs]


def objects_from_dicts(ds) -> tuple[ObjectSpec, ...]:
    return tuple(ObjectSpec(d["name"], d["geom_type"], tuple(d["size"]), tuple(d["rgba"])) for d in ds)


@dataclass
class SceneConfig:
    objects: tuple[ObjectSpec, ...] = OBJECTS
    table_half: float = 0.28  # table is (2*table_half)^2 metres
    wall_half_height: float = 0.012
    pusher_radius: float = 0.025
    pusher_half_height: float = 0.12  # tall => strong occluder
    pusher_cap_radius: float = 0.05  # wider cap on top widens the occluded region
    pusher_cap_half_height: float = 0.03
    # a wide "wrist" at mid-height (no collision) so that a nearby object is fully hidden from an elevated camera
    pusher_wrist_radius: float = 0.07
    pusher_wrist_half_height: float = 0.045
    pusher_wrist_z: float = 0.13
    pusher_rgba: tuple[float, float, float, float] = (0.25, 0.25, 0.28, 1.0)
    # cameras
    n_cameras: int = 4
    camera_azimuths_deg: Sequence[float] | None = None  # default: evenly spaced + offset
    camera_distance: float = 0.85
    camera_elevation_deg: float = 45.0
    camera_fovy_deg: float = 50.0
    image_size: int = 128
    max_depth: float = 2.0  # background depth is clipped to this (metres)
    camera_target_z: float = 0.06  # cameras look slightly above the table so the tall pusher stays in frame
    # physics
    timestep: float = 0.002
    substeps: int = 50  # one env step = substeps * timestep seconds
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    joint_damping: float = 1.0
    object_mass: float = 0.08
    max_action: float = 0.06  # metres of pusher travel per env step (per axis)
    # granular material ("soil"): many small spheres spawned as a pile; 0 disables
    n_grains: int = 0
    grain_radius: float = 0.012
    grain_mass: float = 0.004
    grain_friction: tuple[float, float, float] = (1.2, 0.05, 0.02)  # sliding, torsional, rolling: soil-like, does not roll away
    grain_damping: float = 0.01
    grain_rgba: tuple[float, float, float, float] = (0.55, 0.38, 0.22, 1.0)
    pile_half: float = 0.09  # half-width of the square footprint the pile is spawned in

    def azimuths_deg(self) -> list[float]:
        if self.camera_azimuths_deg is not None:
            assert len(self.camera_azimuths_deg) == self.n_cameras
            return list(self.camera_azimuths_deg)
        # Cameras 0..n-2 evenly spaced; the last one sits halfway between two
        # training cameras so that the held-out viewpoint is an interpolation
        # (the fairest possible test of viewpoint consistency).
        n_train = max(self.n_cameras - 1, 1)
        base = [360.0 * i / n_train for i in range(n_train)]
        held_out = base[0] + 180.0 / n_train
        return base + [held_out] if self.n_cameras > 1 else base

    @property
    def pusher_z(self) -> float:
        return self.pusher_half_height

    @property
    def n_objects(self) -> int:
        return len(self.objects)


def look_at_xyaxes(pos: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Return the MJCF ``xyaxes`` (6,) for a camera at ``pos`` looking at ``target``.

    MuJoCo cameras look down their -z axis with +y up and +x right.
    """
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    z = -fwd
    y = np.cross(z, x)
    return np.concatenate([x, y])


def camera_positions(cfg: SceneConfig) -> np.ndarray:
    """(n_cameras, 3) camera positions on a ring around the table centre."""
    el = np.deg2rad(cfg.camera_elevation_deg)
    out = []
    for az_deg in cfg.azimuths_deg():
        az = np.deg2rad(az_deg)
        out.append(
            [
                cfg.camera_distance * np.cos(el) * np.cos(az),
                cfg.camera_distance * np.cos(el) * np.sin(az),
                cfg.camera_distance * np.sin(el),
            ]
        )
    return np.asarray(out, dtype=float)


def intrinsics(cfg: SceneConfig, image_size: int | None = None) -> np.ndarray:
    """Pinhole intrinsics (3,3) for a square image rendered with ``camera_fovy_deg``."""
    s = image_size or cfg.image_size
    f = 0.5 * s / np.tan(0.5 * np.deg2rad(cfg.camera_fovy_deg))
    return np.array([[f, 0.0, s / 2.0], [0.0, f, s / 2.0], [0.0, 0.0, 1.0]], dtype=float)


def _geom_xml(spec: ObjectSpec) -> str:
    size = " ".join(f"{s:.4f}" for s in spec.size)
    rgba = " ".join(f"{c:.3f}" for c in spec.rgba)
    return f'<geom name="{spec.name}_geom" type="{spec.geom_type}" size="{size}" rgba="{rgba}"/>'


def build_mjcf(cfg: SceneConfig) -> str:
    """Generate the MJCF string for the scene."""
    th = cfg.table_half
    wh = cfg.wall_half_height
    wt = 0.01  # wall thickness (half)
    fr = " ".join(f"{f:g}" for f in cfg.friction)
    prgba = " ".join(f"{c:.3f}" for c in cfg.pusher_rgba)

    bodies = []
    for i, spec in enumerate(cfg.objects):
        z0 = spec.half_height
        bodies.append(
            f'<body name="{spec.name}" pos="{0.1 * i:.3f} 0 {z0:.4f}">'
            f'<joint name="{spec.name}_free" type="free" damping="{cfg.joint_damping:g}"/>'
            f"{_geom_xml(spec)}"
            "</body>"
        )

    grgba = " ".join(f"{c:.3f}" for c in cfg.grain_rgba)
    for i in range(cfg.n_grains):
        bodies.append(
            f'<body name="grain{i}" pos="0 0 {-1.0 - i * 0.05:.3f}">'
            f'<joint name="grain{i}_free" type="free" damping="{cfg.grain_damping:g}"/>'
            f'<geom name="grain{i}_geom" type="sphere" size="{cfg.grain_radius:g}" mass="{cfg.grain_mass:g}" rgba="{grgba}" '
            f'condim="6" friction="{cfg.grain_friction[0]:g} {cfg.grain_friction[1]:g} {cfg.grain_friction[2]:g}"/>'
            "</body>"
        )

    cams = []
    cam_pos = camera_positions(cfg)
    target = np.array([0.0, 0.0, cfg.camera_target_z])
    for i, p in enumerate(cam_pos):
        xy = " ".join(f"{v:.6f}" for v in look_at_xyaxes(p, target))
        ps = " ".join(f"{v:.6f}" for v in p)
        cams.append(f'<camera name="cam{i}" pos="{ps}" xyaxes="{xy}" fovy="{cfg.camera_fovy_deg:g}"/>')

    walls = [
        f'<geom name="wall_px" type="box" size="{wt} {th + wt} {wh}" pos="{th + wt} 0 {wh}" rgba="0.6 0.6 0.62 1"/>',
        f'<geom name="wall_nx" type="box" size="{wt} {th + wt} {wh}" pos="{-th - wt} 0 {wh}" rgba="0.6 0.6 0.62 1"/>',
        f'<geom name="wall_py" type="box" size="{th + wt} {wt} {wh}" pos="0 {th + wt} {wh}" rgba="0.6 0.6 0.62 1"/>',
        f'<geom name="wall_ny" type="box" size="{th + wt} {wt} {wh}" pos="0 {-th - wt} {wh}" rgba="0.6 0.6 0.62 1"/>',
    ]

    xml = f"""
<mujoco model="tabletop_push">
  <option timestep="{cfg.timestep:g}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="{cfg.image_size}" offheight="{cfg.image_size}"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
  </visual>
  <default>
    <geom friction="{fr}" condim="3" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    <default class="object">
      <geom mass="{cfg.object_mass:g}"/>
    </default>
  </default>
  <worldbody>
    <light name="top" pos="0.3 -0.3 2.0" dir="-0.15 0.15 -1" directional="true" castshadow="true"
           diffuse="0.6 0.6 0.6" specular="0.15 0.15 0.15"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.45" rgba="0.33 0.34 0.38 1" contype="0" conaffinity="0"/>
    <geom name="table" type="box" size="{th} {th} 0.02" pos="0 0 -0.02" rgba="0.82 0.80 0.76 1"/>
    {''.join(walls)}
    <body name="pusher" mocap="true" pos="0 0 {cfg.pusher_z:.4f}">
      <geom name="pusher_geom" type="cylinder" size="{cfg.pusher_radius:g} {cfg.pusher_half_height:g}" rgba="{prgba}"/>
      <geom name="pusher_cap" type="cylinder" size="{cfg.pusher_cap_radius:g} {cfg.pusher_cap_half_height:g}"
            pos="0 0 {cfg.pusher_half_height - cfg.pusher_cap_half_height:.4f}" rgba="{prgba}" contype="0" conaffinity="0"/>
      <geom name="pusher_wrist" type="cylinder" size="{cfg.pusher_wrist_radius:g} {cfg.pusher_wrist_half_height:g}"
            pos="0 0 {cfg.pusher_wrist_z - cfg.pusher_z:.4f}" rgba="{prgba}" contype="0" conaffinity="0"/>
    </body>
    {''.join(bodies)}
    {''.join(cams)}
  </worldbody>
</mujoco>
"""
    # apply object class (mass) by string replacement: keep generator simple
    for spec in cfg.objects:
        xml = xml.replace(f'<geom name="{spec.name}_geom"', f'<geom class="object" name="{spec.name}_geom"')
    return xml


def default_camera_split(cfg: SceneConfig) -> tuple[list[int], list[int]]:
    """(train_cams, test_cams): all but the last camera are used for training."""
    if cfg.n_cameras == 1:
        return [0], []
    return list(range(cfg.n_cameras - 1)), [cfg.n_cameras - 1]


def scene_preset(name: str, **overrides) -> SceneConfig:
    """Named scenes used by the experiments.

    * ``default``  : three rigid objects (the main study)
    * ``clutter``  : six rigid objects (locality with more bystanders)
    * ``granular`` : a pile of grains and no rigid objects (scene-changing interaction)
    """
    if name == "default":
        cfg = SceneConfig()
    elif name == "clutter":
        cfg = SceneConfig(objects=CLUTTER_OBJECTS)
    elif name == "granular":
        cfg = SceneConfig(objects=(), n_grains=96, wall_half_height=0.03)
    else:
        raise ValueError(name)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
