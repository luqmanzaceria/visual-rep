# Scope, extensions and where this applies

This document widens the core study in the README. It says what the project
covers, what has been added on top of the original plan, what could come next,
and which real-world settings each property matters for, with a concrete test
for each.

## 1. What the study is, and is not

**In scope.** One question: which properties of a scene representation make an
action-conditioned world model predictable? We hold the predictor, the data and
the readout fixed and change only the representation. Every result is reported
in the same units (centimetres on the table, or grains per cell) and next to a
no-change baseline.

**Out of scope.** Building the best world model, real robots, and policy
learning. Planning is used only as a downstream check that better predictions
lead to better decisions.

**Why the no-change baseline matters.** Most objects in a pushing scene never
move. A model that predicts "nothing happens" therefore looks excellent on
average error, on persistence and on bystander motion. A representation only
earns credit for a property if it keeps that property *while also* moving the
objects that do move. Every table reports both.

## 2. What the first round of experiments showed

These are single-seed results on the held-out evaluation set. Use
`python scripts/summarize.py runs/*` for the current numbers.

| question | evidence | reading |
|---|---|---|
| Does 3D help predict pushes? | Error on moved objects after 4 steps: Gaussians 6.6 cm, DINOv2 8.1, geometry features 10.7, no-change 11.9 | Yes, but every model is still far from accurate |
| Viewpoint consistency | DINOv2's pushed-object error rises from 8.3 cm on training cameras to 14.2 on the held-out camera, worse than no-change. Geometry and Gaussian models stay within about 1 cm | Supported: 2D features do not transfer to a new viewpoint |
| Persistence | Error at reappearance: Gaussians with 3D supervision 0.8 cm. Gaussians trained only on what the camera sees 6.3 cm. DINOv2 9.4 cm | 3D alone is not enough; the training signal must not pull hidden parts toward visible surfaces |
| Locality | Bystander drift after 4 steps: explicit Gaussians 0.4 cm, DINOv2 2.1, geometry features 4.0 | Supported: explicit pieces that move independently keep the rest still |
| Cheap fix for DINOv2 | 4-step rollout training cuts reappearance error from 9.4 to 5.1 cm. Memory tokens add little. Gated updates hurt slightly | The training horizon, not extra architecture, gave most of the gain |
| Planning | Success: simulator planner 85%, best learned model 20%, random 0% | Prediction quality is the bottleneck for decisions |

## 3. Extensions built on top of the original plan

Each extension is implemented, tested, and queued in `scripts/run_ext.sh`.

1. **Visibility-aware observation loss** (`chamfer_visibility=true`). It
   addresses the persistence failure above. Gaussians that the camera cannot see
   in the next frame get no "move to the nearest observed point" gradient. This
   is a first step toward roadmap item 1, a representation designed around the
   properties that matter.
2. **Estimated geometry** (`pointmap_source=mono`). The geometry representation
   so far used simulator depth, which makes it an upper bound. This variant
   estimates depth with Depth Anything V2. It recovers metric scale per frame by
   fitting to the known table plane, which uses camera calibration but no
   per-frame ground truth. Over the 1000 training episodes the median depth
   error is 1.6 cm on the table and 4.4 cm on objects. The run tests whether the
   viewpoint benefit survives realistic depth noise.
3. **Clutter** (`--scene clutter`, six objects). Locality gets harder as more
   bystanders crowd the pushed object. Same metrics, more chances to be wrong.
4. **Scene-changing interaction** (`--scene granular`). This is roadmap item 3
   in miniature. A pile of 96 soil-like grains (rolling friction, a stable
   two-layer mound) is plowed by the pusher, which then steps aside and stops.
   A grid readout predicts grain counts on an 8×8 grid over the table. The
   metrics (`vrep/evaluation/granular.py`) are:
   - **dug**: error on cells that really changed,
   - **spurious**: predicted change in cells far from the pusher's path,
   - **kept**: the fraction of the trench still predicted after the pusher
     stops. 1 means the trench is kept; 0 means the model forgot it or never
     made it. This is persistence of the scene's *state*, not of an object.

## 4. Further ideas, roughly in order of value per effort

- **Seeds and data scaling.** Three seeds per run and a curve of quality
  against data size (250 / 500 / 1000 episodes). This tells us whether the gaps
  between representations close with more data, which is what "outlasts any
  particular backbone" needs to show.
- **Real geometry encoder.** Replace the DINOv2 + point-map stand-in with VGGT
  or DUSt3R features on a GPU machine. `pointmap_source` is the hook.
- **Multi-view Gaussians.** Initialise the scene from all training cameras
  instead of one. This should fix most hidden-object failures at the source and
  isolates how much of persistence is a sensing problem.
- **Object-centric slots.** A fourth representation between patches and
  Gaussians (slot attention over DINOv2 features). It has locality by design but
  no 3D, so it separates the two properties.
- **Camera extrapolation.** The held-out camera now sits between training
  cameras. A camera at a new elevation tests extrapolation, which is the harder
  and more realistic case.
- **Action and embodiment shift.** Train with one pusher shape, test with
  another. Locality claims should hold when the "hand" changes.
- **Planning that uses the properties.** Goals behind occluders (persistence),
  goals in clutter with a penalty for disturbing bystanders (locality), and
  goals seen from a camera the model was not trained on (viewpoint).
- **Granular goals.** "Dig a trench here" and "flatten this pile" as MPC tasks,
  scored on the occupancy grid. This closes the loop for roadmap item 3.
- **Sensor noise and latency.** Blur, colour jitter, depth dropout and
  one-frame action delay. How fast does each representation degrade?

## 5. Where this applies, and how to test it there

Each setting stresses the properties differently. The test column describes a
benchmark that can be built with the tools in this repository, or a short step
beyond them.

| setting | property that matters most | why | test |
|---|---|---|---|
| **Excavation and earthmoving** | Persistence of state, locality | A trench must stay dug after the bucket leaves, and digging here must not move the spoil pile there | Granular scene: high **kept** and low **spurious**. Next: excavation goals as MPC tasks |
| **Construction and assembly** | Locality, persistence | Placing one block must not shift the wall; parts hidden behind the arm must stay where they were put | Clutter scene with stacked objects; bystander drift and reappearance error |
| **Warehouse picking** | Locality in clutter, occlusion by the arm | Bins are crowded and the gripper hides the items beneath it | Clutter scene with sweeps; pushed-object error against bystander drift as object count grows |
| **Agriculture and field robots** | Viewpoint consistency | Cameras move with the vehicle, so every frame is a new viewpoint | Held-out-camera metrics, and the elevation-extrapolation camera above |
| **Fleets of machines** | Viewpoint consistency | A model trained on one machine's camera rig must work on another's | Train on cameras 0–2 and test on camera 3, as now. Then train on one rig height and test on another |
| **Household manipulation** | Persistence | Objects disappear into drawers and behind other objects for many seconds | Longer sweeps with more hidden frames; error as a function of hidden duration |
| **Inspection and mining** | Persistence and viewpoint under noise | Dust, poor light and partial views | Sensor-noise study from section 4 on the granular scene |

## 6. Test matrix

| property | experiment | metric (lower is better unless noted) | code |
|---|---|---|---|
| Persistence (objects) | Sweep episodes: the arm hides an object, then moves on | Error at reappearance; drift while hidden | `persistence_metrics` in `vrep/evaluation/metrics.py` |
| Persistence (scene state) | Plow then stop | **kept** ratio, higher is better | `granular_metrics` in `vrep/evaluation/granular.py` |
| Viewpoint consistency | Every metric on the held-out camera | Gap between held-out and training cameras | `evaluate_all` views |
| Locality (objects) | Push one of three, or one of six, objects | Bystander drift; pushed-object error | `locality_metrics` |
| Locality (granular) | Plow through a pile | **spurious** change far from the path | `granular_metrics` |
| Usefulness | Push-to-goal MPC | Success rate, higher is better; final distance | `vrep/planning/` |
| Sanity | No-change baseline with the same readouts | Must be beaten on moved objects | `scripts/eval.py --baseline copy` |

Unit and smoke tests for all of the above live in `tests/`. Run them with
`python -m pytest`.
