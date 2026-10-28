"""Pre-compute estimated-depth point maps (CPU) for one or more datasets.

Example:
    python scripts/build_mono.py data/push data/eval --threads 8
"""

import argparse

import torch

from vrep.data import EpisodeStore
from vrep.repr.geometry import build_mono_pointmap_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+")
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    for d in args.data:
        store = EpisodeStore(d)
        build_mono_pointmap_cache(store, args.grid, store.root / "features" / f"pointmap_{args.grid}_mono.npy", device=args.device)
        print("done", d)


if __name__ == "__main__":
    main()
