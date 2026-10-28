#!/bin/bash
# Main experiment: train the three representations one after another (GPU jobs
# must not overlap on this machine), evaluate each on the held-out eval set.
#
#   nohup bash scripts/run_main.sh > runs/main.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
mkdir -p runs
STEPS=${STEPS:-6000}
BS=${BS:-32}
EVAL_ARGS="--data data/eval --max_items 400"

run() {
  name=$1; shift
  echo "===== $(date) train $name ====="
  python3 scripts/train.py run=runs/$name steps=$STEPS batch_size=$BS "$@" 2>&1 | grep -v -E "ARB_clip|xFormers|warnings.warn"
  echo "===== $(date) eval $name ====="
  python3 scripts/eval.py runs/$name $EVAL_ARGS 2>&1 | grep -v -E "ARB_clip|xFormers|warnings.warn"
}

run dino         rep=dino
run geo          rep=geo
run gauss_oracle rep=gauss gauss_loss=oracle
run gauss        rep=gauss gauss_loss=chamfer
echo "===== $(date) done ====="
