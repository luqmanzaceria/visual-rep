# What makes a visual representation predictable under action?

16-824 (Visual Learning and Recognition), Fall 2026 project.

General-purpose robots need world models that reliably predict the consequences
of their actions. How well a model predicts depends on how it represents the
scene, and there are no established principles for choosing that representation.

**Hypothesis.** Three properties make physical interaction predictable:

* **Persistence** – the model keeps track of objects while they are hidden from the camera.
* **Viewpoint consistency** – predictions hold up from camera angles the model wasn't trained on.
* **Locality** – an action changes only what it touches; the rest of the scene stays put.

**Approach.** The *same* action-conditioned predictor (`current tokens + action → next tokens`)
is trained on three representations that have these properties to different degrees:

| representation | tokens | what the predictor outputs | module |
|---|---|---|---|
| DINOv2 patch features | 16×16 patches × 384-d, positions = patch grid | next patch features | `vrep/repr/dino.py` |
| Geometry-aware features (DUSt3R/VGGT family, emulated with DINOv2 + per-patch pointmap) | patches × (384 + 3)-d, positions = 3D pointmap | next features + pointmap | `vrep/repr/geometry.py` |
| Dynamic Gaussians | 768 Gaussians × (colour, scale, opacity), positions = 3D centres | 3D displacement per Gaussian | `vrep/repr/gaussians.py`, `vrep/repr/splat.py` |

All three go through one predictor (`vrep/models/predictor.py`, a transformer over tokens with
Fourier positional encoding and an action token) and are compared in one currency:
**object positions on the table**, read out from predicted tokens by a probe
(`vrep/models/probe.py`), plus explicit Gaussian centroids for the 3D representation.

## Environment

A MuJoCo tabletop (`vrep/sim/`): three objects with fixed identity (red cube, green sphere,
blue cylinder), a tall cylindrical pusher that stands in for a robot arm and hides objects behind
it, and a ring of four cameras. Cameras 0–2 are used for training, camera 3 (azimuth halfway
between two training cameras) is held out. Actions are 2D pusher displacements (≤ 6 cm/step).

Scripted episode kinds: `push` (approach one object and push it), `sweep` (pass between a
camera and an object without touching it → occlusion), `random`, `push2` (push, travel, push another).

## Experiments

* **Persistence** (`sweep` episodes): roll out open-loop from just before the occlusion through the
  reappearance; error of the hidden object's position at reappearance vs. a control of visible static objects.
* **Viewpoint consistency**: every metric is computed on the training cameras and on the held-out camera
  (with a readout probe trained on the held-out camera's *real* tokens, so only the predictor is tested).
* **Locality** (`push` episodes): error on the pushed object vs. spurious motion predicted for untouched objects.
* **Planning**: MPC with CEM in token space on push-to-goal tasks (`vrep/planning/`), against an oracle
  planner that uses the true simulator and a random baseline.
* **Final step**: give the cheapest representation (DINOv2) the property that mattered most
  (`n_memory` tokens carried across steps for persistence, `gated` sparse updates with `gate_l1` for locality,
  multi-step `horizon` training) and test whether it closes the gap.

## Extensions

See [docs/SCOPE.md](docs/SCOPE.md) for first results, the extensions, further ideas, and where each property
matters in practice (excavation, construction, warehouses, field robots, fleets), with a test for each.

* `--scene clutter`: six objects, a harder locality test.
* `--scene granular`: a pile of soil-like grains that the pusher plows into a lasting trench (scene-changing
  interaction). It is read out as an 8×8 occupancy grid; metrics in `vrep/evaluation/granular.py`.
* `pointmap_source=mono`: geometry features from estimated depth (Depth Anything V2, calibrated on the table
  plane) instead of simulator depth.
* `chamfer_visibility=true`: Gaussians hidden in the next frame are not pulled toward visible surfaces.
* `scripts/eval.py --baseline copy`: the no-change baseline with the same readouts.

## Usage

```bash
pip install -e .                       # torch, mujoco, einops, omegaconf, ... (see pyproject.toml)
python -m pytest                       # unit + smoke tests (a few minutes on CPU)

python scripts/gen_data.py --out data/push --n 1000 --T 16 --workers 8
python scripts/gen_data.py --out data/eval --n 400 --seed0 5000   # held-out evaluation episodes
python scripts/gen_data.py --scene granular --out data/granular --n 600
python scripts/train.py rep=dino  run=runs/dino
python scripts/train.py rep=geo   run=runs/geo
python scripts/train.py rep=gauss run=runs/gauss
python scripts/eval.py runs/dino --data data/eval   # -> runs/dino/eval_eval.json
python scripts/plan.py runs/dino --n_tasks 20
python scripts/plan.py oracle --n_tasks 20
python scripts/summarize.py runs/dino runs/geo runs/gauss

# full pipeline as run for the report (sequential: GPU jobs must not overlap)
bash scripts/run_main.sh     # dino, geo, gauss_oracle, gauss
bash scripts/run_queue.sh    # decoders, no-change baselines, planning, DINOv2 fixes
python scripts/build_mono.py data/push data/eval   # estimated-depth point maps (CPU)
bash scripts/run_ext.sh      # visibility-aware Gaussians, clutter, granular, estimated geometry
```

Any `TrainConfig` field (`vrep/training/config.py`) can be overridden on the command line, e.g.
`horizon=4 n_memory=8 gated=true gate_l1=0.01`.

## Layout

```
vrep/sim         MuJoCo scene, environment, scripted policies
vrep/data        episode generation and loading
vrep/repr        the three representations (+ differentiable splatter)
vrep/models      shared predictor, readout probe
vrep/training    config, samplers, losses, trainer, rollouts
vrep/evaluation  persistence / viewpoint / locality metrics, granular metrics
vrep/planning    CEM + MPC push-to-goal tasks
scripts/         command-line entry points and experiment queues
docs/            scope, extensions and applications
tests/           pytest suite
```

## Path toward the vision

1. A purpose-built representation designed around the properties that mattered.
2. Closed loop: planning on real tasks, measured by task success.
3. Scene-changing interaction: actions that permanently alter the environment (excavation, construction).
4. Real-world scale: training on real field data across many machines and environments.
