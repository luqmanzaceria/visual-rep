"""Model-predictive control on push-to-goal tasks with a trained run (or the oracle / random baselines).

Example:
    python scripts/plan.py runs/dino --n_tasks 20
    python scripts/plan.py oracle --n_tasks 20
"""

import argparse
import json
from pathlib import Path

import numpy as np

from vrep.planning import ModelCost, OracleCost, RandomCost, make_tasks, run_mpc
from vrep.sim import PushEnv
from vrep.sim.scene import SceneConfig
from vrep.training import load_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run directory, or 'oracle' / 'random'")
    p.add_argument("--n_tasks", type=int, default=20)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=12)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--readout", default="probe", choices=["probe", "explicit"])
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.run in ("oracle", "random"):
        cfg = SceneConfig()
        env = PushEnv(cfg)
        cost = OracleCost(cfg) if args.run == "oracle" else RandomCost()
        name = args.run
    else:
        exp = load_experiment(args.run, device=args.device)
        cfg = exp.store.scene
        env = PushEnv(cfg)
        cost = ModelCost(exp, cam=args.cam, readout=args.readout)
        name = f"{Path(args.run).name}/{args.readout}"
    tasks = make_tasks(env, args.n_tasks)
    results = []
    for i, task in enumerate(tasks):
        r = run_mpc(env, task, cost, horizon=args.horizon, max_steps=args.max_steps, n_samples=args.n_samples, seed=i)
        results.append({k: v for k, v in r.items() if k != "frames"})
        print(f"[{name}] task {i}: success={r['success']} init={r['init_dist'] * 100:.1f}cm final={r['final_dist'] * 100:.1f}cm steps={r['steps']}")
    succ = float(np.mean([r["success"] for r in results]))
    fin = float(np.mean([r["final_dist"] for r in results]))
    summary = {"name": name, "success_rate": succ, "mean_final_dist_cm": fin * 100, "n_tasks": len(tasks), "tasks": results}
    print(f"[{name}] success rate {succ:.2f}, mean final distance {fin * 100:.1f}cm")
    out = Path(args.out) if args.out else (Path(args.run) / f"plan_{args.readout}.json" if args.run not in ("oracle", "random") else Path("runs") / f"plan_{args.run}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
