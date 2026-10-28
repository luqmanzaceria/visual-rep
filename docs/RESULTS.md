# Results: what makes a visual representation predictable under action?

Status as of 28 October 2026. Every number comes from one training seed. Several
runs are still in progress; they are listed at the end.

## In plain terms

We showed a robot's prediction model a table with objects and a "hand" pushing
things around, and asked: after this push, where will everything be? We tried
three ways of describing the scene to the model and checked three abilities.

**1. Seeing from a new angle.** We trained with three cameras and tested with a
fourth.

- The flat-image model, DINOv2, fell apart. On the new camera its push
  predictions were worse than just guessing that nothing moves.
- The two 3D models worked about as well as on the familiar cameras.
- So knowing where things sit in 3D lets predictions carry over to new
  viewpoints. A flat-image model memorises how things look from each camera.

**2. Leaving other objects alone.** When the hand pushes one object, the others
shouldn't move.

- The Gaussian model, which describes the scene as many small 3D blobs it moves
  one by one, barely moved the untouched objects, by about 0.4 cm.
- The flat-image model nudged them about 2 cm.
- So a scene made of separate movable pieces naturally keeps the rest still.

**3. Remembering hidden objects.** The hand passes in front of an object, then
moves away. Does the model still know where the object is?

- Being 3D wasn't enough by itself. It depended on how the model was trained.
- The realistic training method compares predictions with what the camera sees
  next. That taught the model to move hidden blobs onto the hand covering them,
  so the object "teleported".
- When we stopped comparing blobs the camera can't see, the model remembered
  hidden objects well. But it then got worse at predicting pushes. The pushed
  object is usually hidden behind the hand at the moment of contact, so it got
  no training signal either.
- So with only one camera to learn from, the model could learn to remember
  hidden things or to predict pushes, but not both. A run now tests whether
  learning from all three cameras fixes that.

**4. Making the flat-image model better.** We tried giving DINOv2 extra memory,
and separately a rule to leave untouched objects alone. Neither helped much.
What did help was training it to predict several steps ahead instead of one.
That halved its hidden-object errors, but it's still well behind the 3D models.

**The big caveat.** None of the models is good at the main job yet. When an
object really moves 13 cm, the best model is still 6.6 cm off. So when we used
the models to plan pushes to a goal, the best one succeeded only 20% of the
time. A planner using the real physics succeeded 85% of the time.

**In short,** the evidence so far supports two of the three ideas: 3D helps
with new viewpoints, and a scene made of separate pieces keeps untouched objects
still. The third, remembering hidden objects, is more subtle: it depends on
training, not just on 3D. Making the models better at predicting pushes is the
most important next step before trusting the finer comparisons.

## Summary

We trained one action-conditioned predictor on three scene representations and
tested three properties: persistence, viewpoint consistency and locality.

- **Viewpoint consistency is supported.** On a camera never seen in training,
  DINOv2 predicts pushes worse than a model that predicts no motion. The two 3D
  representations stay about as accurate as on the training cameras.
- **Locality is supported.** Gaussians, which the model moves piece by piece,
  keep untouched objects within about 0.4 cm. DINOv2 drifts them about 2 cm and
  the geometry features about 4 cm.
- **Persistence depends on the training signal, not just on 3D.** Gaussians
  keep hidden objects in place only if the loss does not pull hidden parts
  toward visible surfaces. Removing that pull fixes persistence but also removes
  the signal for the pushed object, which is usually hidden behind the pusher
  when it is pushed.
- **The cheap fix for DINOv2 goes about halfway.** Training on 4-step rollouts
  halves its error on hidden objects. Memory tokens and gated updates add
  nothing on top. DINOv2 does not yet match full 3D.
- **No model predicts pushes well yet.** The pushed object moves 13 cm in 4
  steps, and the best model is still 6.6 cm off. In planning, the best model
  reaches 20% of goals, against 85% for a planner that uses the true simulator.

**Viability.** The project is viable as a course project. The pipeline works end
to end, and two of the three properties show clear effects. The main risk is
that the shared predictor captures only about half of each push. The comparison
between representations is therefore made on models that are weak at the core
task, so differences of about 1 cm should not be over-read until there are more
seeds.

## Setup

- **Scene.** A MuJoCo tabletop with three objects (red cube, green sphere, blue
  cylinder) and a tall pusher that stands in for a robot arm. Four cameras ring
  the table. Cameras 0 to 2 are for training; camera 3 sits halfway between two
  of them and is held out.
- **Data.** 1000 training episodes and 400 separate evaluation episodes, each 16
  steps long. Episode types: push one object, sweep the pusher between a camera
  and an object (occlusion), push two objects in turn, and random motion.
- **Representations.**
  - *DINOv2*: 256 patch features per image. 2D, describes only what is visible.
  - *Geometry features*: DINOv2 plus a 3D point per patch from simulator depth.
    This stands in for the DUSt3R/VGGT family and is an upper bound for it.
  - *Gaussians*: 768 3D Gaussians initialised from one camera's RGB-D image. The
    model predicts how each Gaussian moves.
- **Predictor.** The same 4-layer transformer for all three. It takes tokens and
  an action and predicts the next tokens. Training is 6000 steps.
- **Gaussian training signals.**
  - *3D supervision*: the true future positions of each Gaussian. This is an
    upper bound, since a real robot does not observe this.
  - *Single-camera*: match the predicted Gaussians to the points the input
    camera sees in the next frame.
  - *Visibility-aware*: the same, but skip Gaussians the camera cannot see in
    the next frame.
- **Readout.** A small probe reads each object's position off the tokens. It is
  trained on real tokens and applied to predicted ones, so every representation
  is scored in centimetres on the table. Gaussians can also be read directly
  from the centre of each object's Gaussians.
- **Baseline.** A no-change model that predicts nothing moves, scored with the
  same readout. Most objects never move, so this baseline looks strong on
  average error. A representation only gets credit for a property if it keeps
  that property while also moving the objects that do move.

### Metrics

All metrics are in centimetres, and lower is better. Each one compares the
model's predicted displacement of an object with its true displacement, starting
from the same frame.

| metric | what it measures |
|---|---|
| moved@4 | Error after 4 steps, on objects that really moved more than 2 cm |
| pushed@4 | Error after 4 steps on the object being pushed |
| bystander@4 | Predicted motion of objects that did not move, after 4 steps |
| reappear | Error on an object hidden by the pusher, at the frame it reappears |
| drift-hidden | Predicted motion of the hidden object while it is hidden. It really moves only 0.4 cm on average |

The persistence metrics use 71 occlusion events from the training cameras. Each
object is hidden for about 4 frames.

## Main results

Training cameras, 400 evaluation episodes.

| model | moved@4 | pushed@4 | bystander@4 | reappear | drift-hidden |
|---|---|---|---|---|---|
| No-change baseline | 11.9 | 13.0 | 0.0 | 0.4 | 0.0 |
| DINOv2 | 8.1 | 8.3 | 2.1 | 9.4 | 4.4 |
| Geometry features | 10.7 | 10.9 | 4.0 | 12.3 | 6.8 |
| Gaussians, 3D supervision | 8.8 | 9.6 | 1.1 | 0.8 | 0.5 |
| Gaussians, single-camera | 6.6 | 6.6 | 1.3 | 6.3 | 3.8 |
| Gaussians, visibility-aware | 10.5 | 11.1 | 1.2 | 1.4 | 0.7 |

When the Gaussians are read directly instead of through the probe, bystander
drift drops further:

| model | bystander@4 |
|---|---|
| Gaussians, 3D supervision | 0.3 |
| Gaussians, single-camera | 0.4 |
| Gaussians, visibility-aware | 0.3 |

### Viewpoint consistency

The held-out camera's probe is trained on that camera's real tokens, so only the
predictor is being tested.

| model | pushed@4, training cameras | pushed@4, held-out camera | change |
|---|---|---|---|
| No-change baseline | 13.0 | 12.9 | |
| DINOv2 | 8.3 | 14.2 | +5.9 |
| Geometry features | 10.9 | 9.9 | −1.0 |
| Gaussians, 3D supervision | 9.6 | 9.4 | −0.2 |
| Gaussians, single-camera | 6.6 | 7.5 | +0.9 |

DINOv2's push predictions do not transfer to the new viewpoint and end up worse
than predicting nothing. Both 3D representations hold steady.

A second test reuses the training cameras' probe on the held-out camera. The
DINOv2 probe then misplaces objects by 30.7 cm before any prediction, because
its positions are tied to image coordinates. The geometry probe misplaces them
by only 2.6 cm, because its positions are in world coordinates.

### Locality

On push episodes, bystander drift ranks the representations the way the
hypothesis predicts. Explicit Gaussians drift least, at 0.3 to 0.4 cm. DINOv2
drifts 2.1 cm and the geometry features 4.0 cm. The geometry result is
surprising, and it may come from how the point maps are added to the features,
discussed under caveats.

### Persistence

Two results stand out.

1. **3D helps only with the right training signal.** With 3D supervision,
   Gaussians keep a hidden object to within 0.8 cm. With the single-camera
   signal they drift it 3.8 cm while it is hidden. That signal pulls every
   Gaussian toward the nearest visible surface, which for a hidden object is the
   pusher in front of it.
2. **Fixing this costs push accuracy.** The visibility-aware signal brings the
   reappearance error down from 6.3 to 1.4 cm. But the error on moved objects
   rises from 6.6 to 10.5 cm, close to the no-change baseline. The pushed object
   is usually hidden behind the pusher at the moment of contact, so skipping
   hidden Gaussians also skips the pushed object.

This trade-off is a finding in its own right. Under partial observation, a
model can learn to keep hidden objects in place, or to move objects that are
pushed. It cannot learn both from a single camera's supervision. A run
supervising from all three training cameras is in progress to test whether more
views remove the trade-off.

## Final step: giving DINOv2 the missing property

Training cameras.

| model | moved@4 | bystander@4 | reappear | drift-hidden |
|---|---|---|---|---|
| DINOv2 | 8.1 | 2.1 | 9.4 | 4.4 |
| + 4-step rollout training | 8.1 | 1.6 | 5.1 | 3.4 |
| + 4-step training + 8 memory tokens | 8.0 | 1.6 | 4.9 | 3.4 |
| + gated updates (locality) | 8.8 | 2.0 | 9.1 | 4.3 |
| + 4-step training + memory + gates | 8.2 | 1.7 | 6.0 | 3.5 |

Training on longer rollouts did nearly all the work. Memory tokens, the
mechanism meant for persistence, changed little. Gated updates, meant for
locality, slightly hurt. The best DINOv2 variant reaches a reappearance error of
5.1 cm, against 0.8 cm for Gaussians with 3D supervision and 1.4 cm for
visibility-aware Gaussians. The claim that a cheap 2D model can match full 3D is
not supported yet. On cost, a DINOv2 training run took about 7 minutes on the
Mac, against about an hour for a Gaussian run.

## Clutter: six objects

Training on 600 episodes, evaluation on 200, in the same scene with three extra
objects.

| model | view | moved@4 | pushed@4 | bystander@4 | reappear |
|---|---|---|---|---|---|
| DINOv2 | training cameras | 9.4 | 10.8 | 2.2 | 10.2 |
| DINOv2 | held-out camera | 11.8 | 14.3 | 1.5 | 0.7 |
| Geometry features | training cameras | 11.5 | 12.7 | 4.1 | 11.9 |
| Geometry features | held-out camera | 11.0 | 12.5 | 2.8 | 7.3 |

The viewpoint pattern repeats. DINOv2's pushed-object error rises by 3.5 cm on
the held-out camera, while the geometry features stay flat. Push accuracy is
worse than with three objects for both. The held-out camera's persistence
numbers rest on only 6 occlusion events, so they are not meaningful. The
Gaussian run and the no-change baseline for this scene are still in progress.

## Planning

Push one object to a goal 10 to 16 cm away, using model-predictive control with
the cross-entropy method. There are 20 tasks, so each task is worth 5
percentage points.

| planner | success | final distance (cm) |
|---|---|---|
| True simulator (upper bound) | 85% | 3.6 |
| Gaussians, 3D supervision, probe readout | 20% | 10.5 |
| Gaussians, single-camera, direct readout | 20% | 10.4 |
| Gaussians, 3D supervision, direct readout | 15% | 11.3 |
| DINOv2 | 10% | 15.0 |
| Gaussians, visibility-aware, direct readout | 10% | 13.1 |
| Gaussians, single-camera, probe readout | 5% | 12.2 |
| Geometry features | 0% | 11.9 |
| Random actions | 0% | 12.6 |

The goals start 10 to 16 cm from the object. Every learned model ends between
10 and 15 cm from its goal, so none moves the object much closer. The planner
works, since it succeeds with the true simulator. The bottleneck is prediction
quality, which is consistent with the push errors above.

## Caveats

- **One seed.** Differences under about 1 cm, and planning differences under
  about 15 percentage points, could be noise.
- **Weak push prediction.** Every model captures only about half of each push.
  Training was still improving when it stopped at 6000 steps.
- **The geometry stand-in may be unfair.** It is worse than plain DINOv2 on the
  training cameras, even though it has strictly more information. The 3D points
  are appended to the feature vector and predicted as part of it, so errors in
  the predicted points may hurt more than the points help.
- **Simulation only.** Rendering is clean, cameras are perfectly calibrated and
  objects are simple shapes.
- **Gaussians from one view.** Each scene is built from a single camera image,
  so the back of every object is missing. Some persistence failures may come
  from this sensing gap rather than from the representation.

## Still running

These runs are queued on a 4090 server. They will be added here.

| run | question |
|---|---|
| Gaussians, multi-view supervision | Does supervising from all training cameras keep both persistence and pushing? |
| Geometry from estimated depth | Does the viewpoint benefit survive realistic depth? The estimate's median error is 1.6 cm on the table and 4.4 cm on objects |
| Granular plowing: DINOv2, geometry, Gaussians | Can each representation keep a dug trench after the pusher leaves, without disturbing the rest of the pile? |
| Clutter: Gaussians and no-change baseline | Does Gaussian locality hold with six objects? |

## Next steps, in priority order

1. Raise push accuracy before comparing further: train longer and add more push
   episodes.
2. Run three seeds for the main comparison and report the spread.
3. Report every metric next to the no-change baseline, as here.
4. Revisit how the geometry features add their 3D points.

## Reproducing

```bash
python scripts/summarize.py runs/dino runs/geo runs/gauss_oracle runs/gauss runs/gauss_vis \
    runs/dino_h4 runs/dino_mem runs/dino_gate runs/dino_mem_gate
python scripts/summarize.py runs/clutter_dino runs/clutter_geo --eval eval_clutter_eval
```

See the README for the full pipeline and [SCOPE.md](SCOPE.md) for extensions,
ideas and application settings.
