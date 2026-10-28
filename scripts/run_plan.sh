#!/bin/bash
# Planning benchmark: MPC with each trained world model vs. the oracle and random baselines.
#
#   nohup bash scripts/run_plan.sh > runs/plan.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
N=${N:-20}
F="grep -v -E ARB_clip|xFormers|warnings.warn"

python3 scripts/plan.py oracle --n_tasks $N 2>&1 | grep -v -E "ARB_clip|xFormers|warnings.warn"
python3 scripts/plan.py random --n_tasks $N 2>&1 | grep -v -E "ARB_clip|xFormers|warnings.warn"
for run in "$@"; do
  echo "===== $(date) plan $run ====="
  python3 scripts/plan.py runs/$run --n_tasks $N --readout probe 2>&1 | grep -v -E "ARB_clip|xFormers|warnings.warn"
  if [[ $run == gauss* ]]; then
    python3 scripts/plan.py runs/$run --n_tasks $N --readout explicit 2>&1 | grep -v -E "ARB_clip|xFormers|warnings.warn"
  fi
done
echo "===== $(date) done ====="
