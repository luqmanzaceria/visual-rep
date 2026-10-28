"""Goal-reaching push tasks solved by model-predictive control with the learned predictor.

Task: push object ``k`` from its current position to a goal position. The
planner uses the world model as its only knowledge of dynamics: candidate
action sequences are rolled out in token space, and the cost is the distance
between the read-out object position and the goal. The first action is
executed in the real simulator and the process repeats.

Baselines: ``OracleCost`` plans with the true simulator (upper bound for the
planner), ``RandomCost`` ignores the model (lower bound).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from vrep.planning.cem import cem_plan
from vrep.repr.base import TokenBatch
from vrep.sim.env import PushEnv
from vrep.training.builder import Experiment
from vrep.training.rollout import rollout_scene, rollout_tokens


@dataclass
class PushTask:
    seed: int
    target: int
    goal_xy: np.ndarray
    start_state: object  # EnvState


def make_tasks(env: PushEnv, n: int, seed0: int = 1000, dist: tuple[float, float] = (0.10, 0.16)) -> list[PushTask]:
    tasks = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        env.reset(rng)
        k = int(rng.integers(env.n_objects))
        obj = env.object_positions()[k, :2]
        lim = env.cfg.table_half - 0.07
        for _ in range(100):
            ang = rng.uniform(0, 2 * np.pi)
            goal = obj + rng.uniform(*dist) * np.array([np.cos(ang), np.sin(ang)])
            if np.all(np.abs(goal) < lim):
                break
        # start the pusher somewhere not touching anything, within reach
        for _ in range(100):
            p = obj + rng.uniform(0.09, 0.16) * np.array([np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a)])
            if np.all(np.abs(p) < env.cfg.table_half - 0.04):
                break
        env.teleport_pusher(p)
        tasks.append(PushTask(seed0 + i, k, goal.astype(np.float32), env.get_state()))
    return tasks


class ModelCost:
    """Cost of action sequences under the learned world model, from the current observation."""

    def __init__(self, exp: Experiment, cam: int, readout: str = "probe", rng: Optional[np.random.Generator] = None):
        self.exp = exp
        self.cam = cam
        self.readout = readout
        self.rng = rng or np.random.default_rng(0)
        self.tokens: Optional[TokenBatch] = None
        self.scene = None

    @torch.no_grad()
    def observe(self, env: PushEnv) -> None:
        exp = self.exp
        obs = env.render([self.cam])
        if exp.is_gauss:
            self.scene = exp.rep.scene_from_frame(obs["rgb"][0], obs["depth"][0], obs["seg"][0], self.cam, self.rng).to(exp.device)
            self.tokens = exp.rep.tokens(self.scene)
        elif exp.rep.name == "dino":
            self.tokens = exp.rep.tokens_from_rgb(obs["rgb"], exp.device)
        else:
            self.tokens = exp.rep.tokens_from_rgbd(obs["rgb"], obs["depth"], [self.cam], exp.device)

    @torch.no_grad()
    def __call__(self, actions: np.ndarray, target: int, goal_xy: np.ndarray) -> np.ndarray:
        exp = self.exp
        S, H, _ = actions.shape
        a = torch.from_numpy(actions).float().to(exp.device)
        goal = torch.from_numpy(goal_xy).float().to(exp.device)
        if exp.is_gauss:
            scene = self.scene
            scene_b = type(scene)(*[x.expand(S, *x.shape[1:]).contiguous() for x in (scene.mu, scene.color, scene.scale, scene.opacity, scene.group, scene.valid)])
            scenes, _ = rollout_scene(exp.predictor, exp.rep, scene_b, a)
            if self.readout == "explicit":
                c0 = scene.centroids(exp.store.n_objects)[0, target, :2]
                pos = torch.stack([s.centroids(exp.store.n_objects)[:, target, :2] for s in scenes], 1)
                ref = c0[None, None]
                gt_ref = None
            else:
                preds = [exp.rep.tokens(s) for s in scenes]
                pos = torch.stack([exp.probe(p)[0][:, target] for p in preds], 1)
                ref = exp.probe(self.tokens)[0][:, target][:, None]
        else:
            tb = TokenBatch(self.tokens.feat.expand(S, -1, -1).contiguous(), self.tokens.pos.expand(S, -1, -1).contiguous())
            preds, _ = rollout_tokens(exp.predictor, exp.rep, tb, a)
            pos = torch.stack([exp.probe(p)[0][:, target] for p in preds], 1)  # (S,H,2)
            ref = exp.probe(self.tokens)[0][:, target][:, None]
        # cost: final distance to goal (displacement-based to cancel readout bias) + small running term
        cur = torch.from_numpy(self._cur_xy).float().to(exp.device)
        est = cur + (pos - ref)  # estimated object positions along the plan
        d = (est - goal).norm(dim=-1)  # (S,H)
        cost = d[:, -1] + 0.1 * d.mean(1)
        return cost.cpu().numpy()

    def set_current(self, cur_xy: np.ndarray) -> None:
        self._cur_xy = cur_xy


class OracleCost:
    """Plans with the true simulator (an upper bound on what the planner can achieve)."""

    def __init__(self, env_cfg):
        self.env = PushEnv(env_cfg, render=False)

    def observe(self, env: PushEnv) -> None:
        self.state = env.get_state()

    def set_current(self, cur_xy):
        pass

    def __call__(self, actions: np.ndarray, target: int, goal_xy: np.ndarray) -> np.ndarray:
        costs = []
        for seq in actions:
            self.env.set_state(self.state)
            ds = []
            for a in seq:
                self.env.step(a)
                ds.append(np.linalg.norm(self.env.object_positions()[target, :2] - goal_xy))
            costs.append(ds[-1] + 0.1 * np.mean(ds))
        return np.asarray(costs)


class RandomCost:
    def observe(self, env):
        pass

    def set_current(self, cur_xy):
        pass

    def __call__(self, actions, target, goal_xy):
        return np.random.default_rng().uniform(size=actions.shape[0])


def run_mpc(env: PushEnv, task: PushTask, cost, horizon: int = 5, max_steps: int = 12, n_samples: int = 64, n_elite: int = 8,
            n_iters: int = 3, success_thresh: float = 0.04, seed: int = 0, record: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    env.set_state(task.start_state)
    bound = env.cfg.max_action
    dists, frames = [], []
    d0 = float(np.linalg.norm(env.object_positions()[task.target, :2] - task.goal_xy))
    prev = None
    for step in range(max_steps):
        cost.observe(env)
        cur = env.object_positions()[task.target, :2].copy()
        cost.set_current(cur)
        init_mean = None if prev is None else np.concatenate([prev[1:], np.zeros((1, 2))], 0)
        seq, _ = cem_plan(lambda A: cost(A, task.target, task.goal_xy), horizon, 2, bound, rng, n_samples, n_elite, n_iters, init_mean)
        if record:
            frames.append(env.render([0], depth=False, seg=False)["rgb"][0])
        env.step(seq[0])
        prev = seq
        d = float(np.linalg.norm(env.object_positions()[task.target, :2] - task.goal_xy))
        dists.append(d)
        if d < success_thresh:
            break
    final = dists[-1]
    return {"success": final < success_thresh, "final_dist": final, "init_dist": d0, "steps": len(dists), "dists": dists, "frames": frames}
