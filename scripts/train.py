"""Train a predictor on one representation.

Example:
    python scripts/train.py rep=dino run=runs/dino steps=6000
    python scripts/train.py rep=gauss run=runs/gauss gauss_loss=render+chamfer
"""

import sys

from vrep.training import load_config, train


def main():
    cfg = load_config(sys.argv[1:])
    train(cfg)


if __name__ == "__main__":
    main()
