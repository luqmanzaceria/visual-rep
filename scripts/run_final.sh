#!/bin/bash
# Final step: give the cheapest representation (DINOv2) the property that mattered.
#
#   persistence : memory tokens carried across steps + multi-step rollout training
#   locality    : gated per-token updates with an L1 penalty on the gates
#   both        : the two mechanisms together
#
#   nohup bash scripts/run_final.sh > runs/final.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
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

run dino_h4      rep=dino horizon=4                                  # multi-step training alone (control)
run dino_mem     rep=dino horizon=4 n_memory=8                       # + persistence mechanism
run dino_gate    rep=dino gated=true gate_l1=0.02                    # locality mechanism
run dino_mem_gate rep=dino horizon=4 n_memory=8 gated=true gate_l1=0.02
echo "===== $(date) done ====="
