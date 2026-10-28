#!/bin/bash
# Everything that still needs the GPU after run_main.sh, in order.
#   nohup bash scripts/run_queue.sh > runs/queue.log 2>&1 &
cd "$(dirname "$0")/.."
F() { grep -v -E "ARB_clip|xFormers|warnings.warn"; }

echo "===== $(date) decoders ====="
for r in dino geo; do python3 scripts/train_decoder.py runs/$r 2>&1 | F; done

echo "===== $(date) no-change baselines ====="
for r in dino geo gauss; do python3 scripts/eval.py runs/$r --data data/eval --max_items 400 --baseline copy 2>&1 | F; done

echo "===== $(date) planning ====="
bash scripts/run_plan.sh dino geo gauss_oracle gauss

echo "===== $(date) final step ====="
bash scripts/run_final.sh
echo "===== $(date) queue done ====="
