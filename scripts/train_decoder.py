"""(Re)train only the visualisation decoder of an existing dino/geo run.

Example:
    python scripts/train_decoder.py runs/dino [--steps 1500]
"""

import argparse

import numpy as np

from vrep.models import TokenDecoder
from vrep.training import load_experiment
from vrep.training.trainer import train_decoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--steps", type=int, default=1500)
    args = p.parse_args()
    exp = load_experiment(args.run)
    if exp.is_gauss:
        raise SystemExit("Gaussian runs render themselves; no decoder needed")
    grid = exp.rep.encoder.grid if exp.rep.name == "dino" else exp.rep.grid
    exp.decoder = TokenDecoder(exp.rep.feat_dim, grid=grid, out_size=exp.cfg.decoder_size)
    train_decoder(exp, exp.train_eps, exp.store.train_cams, args.steps, np.random.default_rng(exp.cfg.seed))
    exp.save()
    print("saved", exp.cfg.run_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
