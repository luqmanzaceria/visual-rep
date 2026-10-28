#!/bin/bash
# Follow-up to run_ext.sh: multi-view supervised Gaussians (waits for run_ext.sh to finish).
cd "$(dirname "$0")/.."
F() { grep -v -E "ARB_clip|xFormers|warnings.warn|resume_download"; }
while ! grep -q "ext done" runs/ext.log 2>/dev/null; do sleep 120; done
echo "===== $(date) train gauss_mv ====="
python3 scripts/train.py run=runs/gauss_mv rep=gauss gauss_loss=chamfer chamfer_visibility=true multiview_supervision=true steps=6000 batch_size=32 decoder_steps=0 2>&1 | F
python3 scripts/eval.py runs/gauss_mv --data data/eval --max_items 400 2>&1 | F
python3 scripts/plan.py runs/gauss_mv --n_tasks 20 --readout explicit 2>&1 | F | tail -1
echo "===== $(date) ext2 done ====="
