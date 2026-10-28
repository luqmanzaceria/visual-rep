#!/bin/bash
# Extension experiments (run after run_main.sh / run_queue.sh), sequential on the GPU.
#   nohup bash scripts/run_ext.sh > runs/ext.log 2>&1 &
cd "$(dirname "$0")/.."
F() { grep -v -E "ARB_clip|xFormers|warnings.warn|resume_download"; }
STEPS=${STEPS:-6000}
COMMON="steps=$STEPS batch_size=32 decoder_steps=0"

run() {  # run <name> <data> <eval_data> args...
  name=$1; data=$2; evald=$3; shift 3
  echo "===== $(date) train $name ====="
  python3 scripts/train.py run=runs/$name data=$data $COMMON "$@" 2>&1 | F
  echo "===== $(date) eval $name ====="
  python3 scripts/eval.py runs/$name --data $evald --max_items 400 2>&1 | F
}
copy() {  # copy <name> <eval_data>
  python3 scripts/eval.py runs/$1 --data $2 --max_items 400 --baseline copy 2>&1 | F
}

# 1. purpose-built fix for persistence: visibility-aware observation loss for Gaussians
run gauss_vis data/push data/eval rep=gauss gauss_loss=chamfer chamfer_visibility=true
python3 scripts/plan.py runs/gauss_vis --n_tasks 20 --readout explicit 2>&1 | F | tail -1

# 2. clutter: locality with six objects
run clutter_dino  data/clutter data/clutter_eval rep=dino
run clutter_geo   data/clutter data/clutter_eval rep=geo
run clutter_gauss data/clutter data/clutter_eval rep=gauss gauss_loss=chamfer chamfer_visibility=true
copy clutter_dino data/clutter_eval

# 3. scene-changing interaction: plowing granular material
run granular_dino  data/granular data/granular_eval rep=dino
run granular_geo   data/granular data/granular_eval rep=geo
run granular_gauss data/granular data/granular_eval rep=gauss gauss_loss=chamfer chamfer_visibility=true
copy granular_dino data/granular_eval

# 4. realistic geometry: estimated (monocular) point maps instead of simulator depth
while [ ! -f data/eval/features/pointmap_16_mono.json ]; do sleep 120; done
run geo_mono data/push data/eval rep=geo pointmap_source=mono
python3 scripts/plan.py runs/geo_mono --n_tasks 20 2>&1 | F | tail -1
echo "===== $(date) ext done ====="
