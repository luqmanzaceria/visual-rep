"""Generate a pushing dataset.

Example:
    python scripts/gen_data.py --out data/push --n 1000 --T 16 --workers 8
"""

import argparse

from vrep.data import generate_dataset
from vrep.sim.scene import scene_preset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/push")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--kind", default=None, help="restrict to one episode kind (push|sweep|random|push2|plow)")
    p.add_argument("--scene", default="default", choices=["default", "clutter", "granular"])
    args = p.parse_args()
    cfg = scene_preset(args.scene, image_size=args.image_size)
    out = generate_dataset(args.out, args.n, T=args.T, seed0=args.seed0, cfg=cfg, workers=args.workers, kind=args.kind)
    print("wrote", out)


if __name__ == "__main__":
    main()
