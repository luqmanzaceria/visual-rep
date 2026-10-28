#!/bin/bash
# Single-GPU queue for the shared 4090 server, ordered by how much each run tells us.
#   CUDA_VISIBLE_DEVICES=7 nohup bash scripts/remote_queue.sh > runs/remote_queue.log 2>&1 &
source "$(dirname "$0")/remote_env.sh"
F() { grep -v -E "ARB_clip|xFormers|warnings.warn|resume_download"; }
C="steps=6000 batch_size=32 decoder_steps=0"
run() { n=$1; d=$2; e=$3; shift 3
  if [ -f runs/$n/eval_$(basename $e).json ]; then echo "skip $n (done)"; return; fi
  echo "===== $(date) train $n ====="; python scripts/train.py run=runs/$n data=$d $C "$@" 2>&1 | F
  echo "===== $(date) eval $n ====="; python scripts/eval.py runs/$n --data $e --max_items 400 2>&1 | F; }

# 1. does multi-view supervision keep both persistence and pushing?
run gauss_mv data/push data/eval rep=gauss gauss_loss=chamfer chamfer_visibility=true multiview_supervision=true
python scripts/plan.py runs/gauss_mv --n_tasks 20 --readout explicit 2>&1 | F | tail -1

# 2. scene-changing interaction
run granular_dino  data/granular data/granular_eval rep=dino
run granular_geo   data/granular data/granular_eval rep=geo
run granular_gauss data/granular data/granular_eval rep=gauss gauss_loss=chamfer chamfer_visibility=true
python scripts/eval.py runs/granular_dino --data data/granular_eval --max_items 400 --baseline copy 2>&1 | F

# 3. realistic geometry from estimated depth
echo "===== $(date) mono depth point maps ====="
python scripts/build_mono.py data/push data/eval 2>&1 | F | grep -E "mono pointmap|done|Error"
run geo_mono data/push data/eval rep=geo pointmap_source=mono
python scripts/plan.py runs/geo_mono --n_tasks 20 2>&1 | F | tail -1

# 4. clutter
run clutter_gauss data/clutter data/clutter_eval rep=gauss gauss_loss=chamfer chamfer_visibility=true
python scripts/eval.py runs/clutter_dino --data data/clutter_eval --max_items 400 --baseline copy 2>&1 | F
echo "===== $(date) remote queue done ====="
