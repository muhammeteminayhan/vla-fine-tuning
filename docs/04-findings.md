# Phase 4 — Results and Honest Reading

**Date:** 2026-09-22
**Resolution:** 10 episodes per task, the published LIBERO protocol, over three
training seeds — 300 evaluation episodes per point.
**Status:** complete.

Every number traces to a file under `results/`; `results/INDEX.md` is the map.

---

## 1. The teaching cost curve

`results/teaching_cost_curve.json` · figure `assets/teaching_cost_curve.png`

| K | operator min / new task | success | Wilson 95% | per-seed |
|---|---|---|---|---|
| 0 | 0.0 | 0/100 = 0.0% | [0.0%, 3.7%] | — |
| 5 | 2.3 | 200/300 = 66.7% | [61.2%, 71.8%] | 61%, 64%, 75% |
| 10 | 4.6 | 201/300 = 67.0% | [61.5%, 72.1%] | 62%, 68%, 71% |
| 20 | 9.1 | 219/300 = 73.0% | [67.7%, 77.7%] | 64%, 75%, 80% |
| 40 | 18.2 | 229/300 = 76.3% | [71.2%, 80.8%] | 71%, 73%, 85% |

**Almost all of the value arrives in the first five demonstrations.** The
increments are +66.7 points from zero to five, then **+0.3**, then +6.0, then
+3.3. K=5 and K=10 are indistinguishable — their intervals very nearly coincide.
Going from five demonstrations to forty costs the operator eight times as much
time and buys 9.6 points.

If a plant is deciding how much demonstration a new part needs, this data says
*a handful*, and says it with 300 episodes per point rather than an anecdote.

The curve rises monotonically, but only the K=5 → K=20 and K=5 → K=40 steps are
separated by their intervals. The rest is within sampling noise.

---

## 2. The finding that most qualifies the claim: teaching erases the old line

`results/forgetting.json` · figure `assets/forgetting.png`
(measured at 5 episodes per task, 50 episodes per bar)

Evaluated on `libero_spatial`, one of the three suites stage 1 was trained on:

| model | libero_spatial (old line) | libero_object (new part) |
|---|---|---|
| stage 1, before teaching | 19/50 = **38%** [25.9%, 51.8%] | 0% |
| + 5 demonstrations of a new part | 0/50 = **0%** [0.0%, 7.1%] | 58% |
| + 40 demonstrations of a new part | 0/50 = **0%** [0.0%, 7.1%] | 82% |

**Teaching one new part destroys the old capability completely**, and five
demonstrations are enough to do it. The intervals are disjoint.

This follows from the design rather than contradicting it. Stage 2 continues
training the *same* LoRA adapter — LeRobot logs "PEFT adapter already loaded from
checkpoint, skipping wrap_with_peft" — at lr=1e-3, for 6000 steps, on nothing but
the new task's data. The adapter is overwritten.

**What it does to the business argument.** The headline survives: a new part
costs roughly two operator-minutes of demonstration rather than an integrator's
invoice. But it survives only with this attached — *the cell forgets its previous
work in the process*. A practical answer exists and LoRA makes it cheap: each
adapter is **11.9 MB**, so a plant could keep one per part and load the right one.
That is a different product from "one model that knows the whole line", and
saying so is the difference between a result and a sales pitch.

---

## 3. Stage 1 is undertrained, which limits what the scenario models

The stage-1 model scores **38%** on the suite it was trained on. It is the most
undertrained run in the study: the step budget was fixed at 6000 for every run,
while stage 1's dataset is 3.5x larger than K=40's.

| run | frames | epochs at 96,000 samples |
|---|---|---|
| stage 1 | 206,481 | **0.47** |
| K=40 | 59,017 | 1.63 |
| K=5 | 7,377 | 13.0 |

The curve is unaffected — the K=0 reference is stage 1 on `libero_object`,
measured at 0/100 — and the model genuinely cannot do the new parts. But the
framing is weaker than intended. Accurate wording: stage 1 gave the model
familiarity with LIBERO-style manipulation; it did not make it competent at its
own suite. Fixing it means retraining stage 1 and all twelve K-shot runs from the
new base, roughly 18 hours, which was not spent.

---

## 4. Failures are not near-misses

`results/failure_analysis.json`

| | n | mean steps |
|---|---|---|
| successes | 849 | 141.6 (median 137) |
| failures | 351 | 280.0 — all at the cap |

The environment terminates early only on success, so failures reaching the cap
is expected. The informative part is the ratio: a typical success needs 137 steps
against a 280-step budget, so **every failure had about twice the time a success
needs**. A larger step budget would not convert them.

Failure rate falls with K, and the fall is small after the first jump:

| K | failure rate |
|---|---|
| 5 | 100/300 = 33.3% |
| 10 | 99/300 = 33.0% |
| 20 | 81/300 = 27.0% |
| 40 | 71/300 = 23.7% |

---

## 5. Task difficulty is real, not noise

`results/task_difficulty.json`, pooled over all twelve runs, 120 episodes per task.

| task | success | Wilson 95% |
|---|---|---|
| cream cheese | 109/120 = 91% | [84.3%, 94.8%] |
| salad dressing | 104/120 = 87% | [79.4%, 91.6%] |
| tomato sauce | 103/120 = 86% | [78.5%, 91.0%] |
| chocolate pudding | 91/120 = 76% | [67.4%, 82.6%] |
| orange juice | 85/120 = 71% | [62.2%, 78.2%] |
| alphabet soup | 85/120 = 71% | [62.2%, 78.2%] |
| butter | 83/120 = 69% | [60.4%, 76.7%] |
| ketchup | 75/120 = 62% | [53.6%, 70.6%] |
| bbq sauce | 63/120 = 52% | [43.6%, 61.2%] |
| **milk** | 51/120 = **42%** | [34.0%, 51.4%] |

48 points between hardest and easiest, with disjoint intervals: difficulty is a
real effect. Pooling mixes K, so this answers "which tasks are hard for this
method", not "how difficulty changes with K".

A caution about reading rankings from smaller samples: the 5-episode sweep put
bbq sauce hardest at 40% and tomato sauce easiest at 96%. At 45 episodes per task
those intervals were wide enough that the ordering was not reliable, and the
10-episode sweep — a different sample, see §6 — reorders the middle of the table
while keeping milk and bbq sauce at the bottom.

---

## 6. What could not be measured, and why

**Failure modes are not classified.** CLAUDE.md asks for failures to be sorted
into wrong object, missed grasp, early release and missed target. Two attempts,
both recorded rather than quietly dropped:

1. *From the rendered video.* The eval video is the agentview only, 360×360
   (`LiberoEnv.render` returns the first camera in the observation dict), with no
   gripper state. Cropped frame strips (`src/analysis/failure_strips.py`) show the
   arm reaching the object cluster and leaving with the basket still empty, in
   every seed — consistent with grasp failure, but that is an impression from
   eight frames, not a measurement.

2. *From a recorded rollout dataset.* `--eval.recording=true` would capture
   `eef_pos`, `gripper_qpos` **and the wrist camera**, which is exactly what the
   video lacks. It fails upstream:

   ```
   ValueError: Feature names should not contain '/'. Found '/' in
   'robot_state/eef/pos', 'robot_state/gripper/qpos', 'pixels/agentview_image', ...
   ```

   LIBERO's feature keys are nested with `/`, and LeRobot's dataset validator
   (`utils/feature_utils.py:44`) rejects them, so recording is unusable for this
   environment at commit `2774d9bd`. Classifying failure modes properly needs
   either that fixed upstream or a custom rollout loop.

**Single benchmark, single model, simulation only.** No real hardware, no
sim-to-real measurement, one policy architecture, one benchmark suite, one
held-out suite of ten tasks.

**The operator-time axis is mostly an assumption.** 7.4 s of demonstration is
measured; 20 s of reset per demonstration is assumed and cannot be measured from
this dataset. At that value the assumption is 73% of the figure. Changing it
rescales the axis without changing the curve's shape or where it flattens —
`docs/02-experiment-design.md` §4 carries the sensitivity table.
