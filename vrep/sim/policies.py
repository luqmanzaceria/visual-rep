"""Scripted pusher behaviours used to generate training / evaluation episodes.

Every behaviour is expressed as a *plan*: an initial pusher placement (applied by
teleport before the first frame is recorded) and a function that emits the next
action given the current environment. Episodes therefore stay physically
consistent after frame 0: all later motion comes through ``env.step``.

Behaviours
----------
* ``push``   : approach one object from a random side and push through it.
* ``sweep``  : move between a chosen camera and an object so that the pusher
               hides the object from that camera for a few frames, then moves on
               (the persistence experiment). The sweep keeps a gap to the object
               so that it usually does not touch it.
* ``random`` : a smooth random walk (coverage of free-space motion).
* ``push2``  : push one object, then travel to a second object and push it
               (covers "move without contact" followed by contact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from vrep.sim.env import PushEnv

EPISODE_KINDS = ("push", "sweep", "random", "push2")
GRANULAR_KINDS = ("plow", "random")


@dataclass
class Plan:
    kind: str
    start_xy: np.ndarray  # pusher placement before frame 0
    policy: Callable[[PushEnv, int], np.ndarray]  # (env, t) -> action
    target: int = -1  # object index the plan is about (-1: none)
    cam: int = -1  # camera the plan is about (sweeps only)
    meta: dict = field(default_factory=dict)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _clip_table(env: PushEnv, xy: np.ndarray, margin: float = 0.03) -> np.ndarray:
    lim = env.cfg.table_half - margin
    return np.clip(xy, -lim, lim)


def _approach_point(env: PushEnv, k: int, direction: np.ndarray, gap: float) -> np.ndarray:
    """Point at ``gap`` metres from object ``k``'s surface, on the side opposite to ``direction``."""
    obj = env.object_positions()[k, :2]
    r = env.cfg.objects[k].radius + env.cfg.pusher_radius + gap
    return obj - direction * r


def _free_of_objects(env: PushEnv, xy: np.ndarray, extra: float = 0.01) -> bool:
    gp = env.grain_positions()
    if len(gp) and np.min(np.linalg.norm(gp[:, :2] - xy, axis=1)) < env.cfg.grain_radius + env.cfg.pusher_radius + extra:
        return False
    pos = env.object_positions()[:, :2]
    for k, spec in enumerate(env.cfg.objects):
        if np.linalg.norm(pos[k] - xy) < spec.radius + env.cfg.pusher_radius + extra:
            return False
    return True


def _go_to(env: PushEnv, goal: np.ndarray, speed: float) -> np.ndarray:
    """One action that moves the pusher toward ``goal`` at most ``speed``."""
    cur = env.pusher_position()[:2]
    d = goal - cur
    n = np.linalg.norm(d)
    if n <= speed:
        return d
    return d / n * speed


# ---------------------------------------------------------------------- plans
def plan_push(env: PushEnv, rng: np.random.Generator, target: Optional[int] = None) -> Plan:
    cfg = env.cfg
    k = int(rng.integers(env.n_objects)) if target is None else target
    for _ in range(50):
        ang = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(ang), np.sin(ang)])
        start = _approach_point(env, k, direction, gap=rng.uniform(0.01, 0.04))
        if np.all(np.abs(start) < cfg.table_half - 0.03) and _free_of_objects(env, start):
            break
    speed = rng.uniform(0.6, 1.0) * cfg.max_action
    noise = rng.uniform(0.0, 0.15)
    curve = rng.uniform(-0.08, 0.08)  # slow rotation of the push direction

    def policy(env: PushEnv, t: int) -> np.ndarray:
        th = np.arctan2(direction[1], direction[0]) + curve * t
        d = np.array([np.cos(th), np.sin(th)])
        a = d * speed + rng.normal(0, noise * speed, size=2)
        return a

    return Plan("push", _clip_table(env, start), policy, target=k, meta={"direction": direction.tolist()})


def plan_sweep(env: PushEnv, rng: np.random.Generator, target: Optional[int] = None, cam: Optional[int] = None) -> Plan:
    """Sweep the pusher between camera ``cam`` and object ``target`` without touching it."""
    cfg = env.cfg
    k = int(rng.integers(env.n_objects)) if target is None else target
    c = int(rng.integers(cfg.n_cameras)) if cam is None else cam
    _, cam_t = env.camera_extrinsics(c)
    obj = env.object_positions()[k, :2]
    to_cam = _unit(cam_t[:2] - obj)  # direction from object toward camera (table plane)
    perp = np.array([-to_cam[1], to_cam[0]])
    gap = rng.uniform(0.015, 0.05)
    r = cfg.objects[k].radius + cfg.pusher_radius + gap
    centre = obj + to_cam * r  # point between the camera and the object
    half_len = rng.uniform(0.10, 0.18)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    start = _clip_table(env, centre - sign * perp * half_len)
    end = _clip_table(env, centre + sign * perp * half_len)
    speed = rng.uniform(0.45, 0.8) * cfg.max_action
    wobble = rng.uniform(0.0, 0.1)
    state = {"phase": 0}

    def policy(env: PushEnv, t: int) -> np.ndarray:
        cur = env.pusher_position()[:2]
        if state["phase"] == 0 and np.linalg.norm(end - cur) < 1e-3:
            state["phase"] = 1
        goal = end if state["phase"] == 0 else start
        a = _go_to(env, goal, speed)
        return a + rng.normal(0, wobble * speed, size=2)

    return Plan("sweep", start, policy, target=k, cam=c, meta={"end": end.tolist()})


def plan_random(env: PushEnv, rng: np.random.Generator) -> Plan:
    cfg = env.cfg
    for _ in range(50):
        start = rng.uniform(-cfg.table_half + 0.04, cfg.table_half - 0.04, size=2)
        if _free_of_objects(env, start):
            break
    vel = rng.normal(0, 0.4, size=2) * cfg.max_action
    state = {"v": vel}

    def policy(env: PushEnv, t: int) -> np.ndarray:
        v = 0.8 * state["v"] + rng.normal(0, 0.35, size=2) * cfg.max_action
        v = np.clip(v, -cfg.max_action, cfg.max_action)
        cur = env.pusher_position()[:2]
        # bounce off the table edges
        lim = cfg.table_half - 0.05
        for i in range(2):
            if (cur[i] > lim and v[i] > 0) or (cur[i] < -lim and v[i] < 0):
                v[i] = -v[i]
        state["v"] = v
        return v

    return Plan("random", start, policy)


def plan_push2(env: PushEnv, rng: np.random.Generator, n_first: int = 6) -> Plan:
    """Push one object for ``n_first`` steps, then travel to and push another one."""
    cfg = env.cfg
    k1 = int(rng.integers(env.n_objects))
    k2 = int((k1 + 1 + rng.integers(env.n_objects - 1)) % env.n_objects)
    first = plan_push(env, rng, target=k1)
    speed = cfg.max_action
    state = {"phase": 0, "approach": None, "dir": None, "n_push": 0}

    def policy(env: PushEnv, t: int) -> np.ndarray:
        if t < n_first:
            return first.policy(env, t)
        if state["phase"] == 0:
            # choose an approach point for the second object that avoids the pusher passing through it
            obj = env.object_positions()[k2, :2]
            cur = env.pusher_position()[:2]
            d = _unit(obj - cur)
            state["dir"] = d
            state["approach"] = _clip_table(env, _approach_point(env, k2, d, gap=0.02))
            state["phase"] = 1
        if state["phase"] == 1:
            a = _go_to(env, state["approach"], speed)
            if np.linalg.norm(state["approach"] - env.pusher_position()[:2]) <= speed + 1e-6:
                state["phase"] = 2
            return a
        return state["dir"] * 0.8 * cfg.max_action + rng.normal(0, 0.05 * cfg.max_action, size=2)

    return Plan("push2", first.start_xy, policy, target=k1, meta={"second": k2})


def plan_plow(env: PushEnv, rng: np.random.Generator) -> Plan:
    """Drive straight through the granular pile, step aside, then stop.

    The pusher's path through the pile carves a trench that stays after the
    pusher leaves; the final stationary frames test whether a model keeps the
    change (permanence) instead of drifting back to the initial pile.
    """
    cfg = env.cfg
    g = env.grain_positions()[:, :2]
    centre = g.mean(0) + rng.uniform(-0.3, 0.3, size=2) * cfg.pile_half
    ang = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(ang), np.sin(ang)])
    reach = cfg.pile_half + cfg.pusher_radius + 0.03
    start = _clip_table(env, centre - d * reach)
    through = _clip_table(env, centre + d * (reach + rng.uniform(0.0, 0.04)))
    side = np.array([-d[1], d[0]]) * (1.0 if rng.uniform() < 0.5 else -1.0)
    away = _clip_table(env, through + side * rng.uniform(0.08, 0.14))
    speed = rng.uniform(0.5, 0.8) * cfg.max_action
    state = {"phase": 0}

    def policy(env: PushEnv, t: int) -> np.ndarray:
        cur = env.pusher_position()[:2]
        if state["phase"] == 0:
            if np.linalg.norm(through - cur) < 1e-3:
                state["phase"] = 1
            else:
                return _go_to(env, through, speed)
        if state["phase"] == 1:
            if np.linalg.norm(away - cur) < 1e-3:
                state["phase"] = 2
            else:
                return _go_to(env, away, speed)
        return np.zeros(2)

    return Plan("plow", start, policy, meta={"centre": centre.tolist(), "direction": d.tolist()})


def make_plan(env: PushEnv, rng: np.random.Generator, kind: str | None = None, probs=(0.45, 0.30, 0.10, 0.15)) -> Plan:
    if env.cfg.n_grains > 0 and env.n_objects == 0:
        if kind is None:
            kind = "plow" if rng.uniform() < 0.8 else "random"
        if kind == "plow":
            return plan_plow(env, rng)
        if kind == "random":
            return plan_random(env, rng)
        raise ValueError(f"{kind} is not available in a granular scene without objects")
    if kind is None:
        kind = EPISODE_KINDS[rng.choice(len(EPISODE_KINDS), p=np.asarray(probs) / np.sum(probs))]
    if kind == "push":
        return plan_push(env, rng)
    if kind == "sweep":
        return plan_sweep(env, rng)
    if kind == "random":
        return plan_random(env, rng)
    if kind == "push2":
        return plan_push2(env, rng)
    raise ValueError(kind)
