"""Evaluate a trained run on the three properties.

Example:
    python scripts/eval.py runs/dino [--horizon 8] [--max_items 400]
"""

import argparse
from pathlib import Path

from vrep.evaluation.metrics import evaluate_all, save_results
from vrep.training import load_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--max_items", type=int, default=400)
    p.add_argument("--device", default=None)
    p.add_argument("--data", default=None, help="evaluate on a different dataset (all of its episodes are used)")
    p.add_argument("--out", default=None)
    p.add_argument("--baseline", default=None, choices=[None, "copy"], help="'copy': evaluate the no-change baseline with this run's readouts")
    args = p.parse_args()
    exp = load_experiment(args.run, device=args.device, data=args.data)
    episodes = list(range(len(exp.store))) if args.data else None
    results = evaluate_all(exp, horizon=args.horizon, max_items=args.max_items, episodes=episodes, baseline=args.baseline)
    suffix = ("" if not args.data else f"_{Path(args.data).name}") + ("" if not args.baseline else f"_{args.baseline}")
    out = Path(args.out) if args.out else Path(args.run) / f"eval{suffix}.json"
    save_results(results, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
