# Phase 4 — Results and Honest Reading

**Date:** 2026-09-21
**Status:** in progress. The curve and the findings below are final at 5
episodes per task; a re-evaluation at the published 10-per-task resolution is
running.

Every number traces to a file under `results/`.

---

## 1. The teaching cost curve

`results/teaching_cost_curve.json`, figure `assets/teaching_cost_curve.png`.
Three training seeds, 150 evaluation episodes per point.

| K | operator min / new task | success | Wilson 95% | per-seed |
|---|---|---|---|---|
| 0 | 0 | 0/50 = 0% | [0.0%, 7.1%] | — |
| 5 | 2.3 | 98/150 = 65.3% | [57.4%, 72.5%] | 58%, 68%, 70% |
| 10 | 4.6 | 106/150 = 70.7% | [62.9%, 77.4%] | 68%, 72%, 72% |
| 20 | 9.1 | 113/150 = 75.3% | [67.9%, 81.5%] | 64%, 76%, 86% |
| 40 | 18.2 | 118/150 = 78.7% | [71.4%, 84.5%] | 72%, 82%, 82% |

**The curve saturates early.** Increments are +65 points from 0 to 5
demonstrations, then +5.4, +4.6 and +3.4. Doubling the operator's time from 9 to
18 minutes per part buys 3.4 points, and the K=5 and K=40 intervals almost
touch. If the question is "how much demonstration does a new part need", the
answer this data supports is **a handful, and the rest is diminishing returns**.

---

## 2. The finding that most qualifies the claim: teaching erases the old line

`results/forgetting_*_spatial/results.json`.

Evaluated on `libero_spatial`, one of the three suites stage 1 was trained on:

| model | libero_spatial (old line) | libero_object (new part) |
|---|---|---|
| stage 1, before teaching | 19/50 = **38%** [25.9%, 51.8%] | 0% |
| + K=5 of the new part | 0/50 = **0%** [0.0%, 7.1%] | 58% |
| + K=40 of the new part | 0/50 = **0%** [0.0%, 7.1%] | 82% |

**Teaching one new part destroys the old capability completely**, and five
demonstrations are enough to do it. The intervals are disjoint, so this is not a
measurement artefact.

It is also a direct consequence of the design rather than a surprise. Stage 2
continues training the *same* LoRA adapter (verified in Phase 3: LeRobot logs
"PEFT adapter already loaded from checkpoint, skipping wrap_with_peft"), at
lr=1e-3, for 6000 steps, on nothing but the new task's data. The adapter is
simply overwritten.

**What this does to the business argument.** The headline — a new part costs
about two operator-minutes of demonstration instead of an integrator's invoice —
survives, but only with this attached: *the cell forgets its previous work in
the process.* A practical answer exists, and LoRA makes it cheap: each adapter
is **11.9 MB**, so a plant could keep one adapter per part and load the right
one. But that is a different product from "one model that knows the whole line",
and the report must say so rather than quietly imply the latter.

Measured on one seed per condition. Given 0/50 in both, one seed is enough to
establish the effect, not its precise size.

---

## 3. Stage 1 is undertrained, and that limits what the K=0 line means

The stage-1 model scores **38%** on the suite it was trained on. It is the most
undertrained run in the study: the step budget was fixed at 6000 for every run,
but stage 1's dataset is 3.5x larger than K=40's.

| run | frames | epochs at 96,000 samples |
|---|---|---|
| stage 1 | 206,481 | **0.47** |
| K=40 | 59,017 | 1.63 |
| K=5 | 7,377 | 13.0 |

This does not affect the curve — the K=0 reference is "stage 1 on
`libero_object`", measured at 0/50, and the model genuinely cannot do the new
parts. But the scenario framing is weaker than intended. Honest wording: stage 1
gave the model familiarity with LIBERO-style manipulation; it did not make it
competent at its own suite.

---

## 4. Failures are not near-misses

`results/failure_analysis.json`.

| | n | mean steps |
|---|---|---|
| successes | 435 | 139.9 (median 138) |
| failures | 165 | 280.0 — all at the cap |

The environment only terminates early on success, so failures reaching the cap
is expected. The informative part is the ratio: a typical success needs 138
steps against a 280-step budget, so **every failure had about twice the time a
success needs**. A larger step budget would not convert them.

Failure rate falls steadily with K — 34.7%, 29.3%, 24.7%, 21.3% at K = 5, 10,
20, 40 — and concentrates heavily on two tasks.

---

## 5. Task difficulty is real, not noise

`results/task_difficulty.json`, pooled over all 12 runs, 45 episodes per task.

| task | success | Wilson 95% |
|---|---|---|
| tomato sauce | 43/45 = 96% | [85.2%, 98.8%] |
| salad dressing | 41/45 = 91% | [79.3%, 96.5%] |
| chocolate pudding / cream cheese / ketchup | 35/45 = 78% | [63.7%, 87.5%] |
| orange juice | 33/45 = 73% | [59.0%, 84.0%] |
| butter / alphabet soup | 31/45 = 69% | [54.3%, 80.5%] |
| milk | 23/45 = 51% | [37.0%, 65.0%] |
| **bbq sauce** | 18/45 = **40%** | [27.0%, 54.5%] |

56 points between hardest and easiest, with disjoint intervals. Per-K, per-seed
task rates rest on 10-15 episodes where the interval is ~50 points wide, so only
the pooled view supports any ranking; the cost is that pooling mixes K.

The same episodes fail across independent training seeds — task 3's episodes 1
and 3 fail in all three K=40 runs — which points at the initial scene
configuration rather than at training randomness.

---

## 6. What could not be measured, and why

**Failure modes are not classified.** CLAUDE.md asks for failures to be sorted
into wrong object, missed grasp, early release and missed target. That cannot be
done reliably from what this harness keeps: the eval video is the agentview
only, 360x360 (`LiberoEnv.render` returns the first camera in the observation
dict), with no gripper state and no object poses recorded.

Failure strips for the two hardest tasks show the arm reaching the object
cluster and leaving with the basket still empty, in every seed, which is
consistent with grasp failure. That is an impression from eight frames, not a
measurement, and it is recorded as such. A rigorous classification needs the
harness to log object poses or the wrist camera, which would mean re-running the
evaluations.

**Single benchmark, single model, simulation only.** No real hardware, no
sim-to-real measurement, one policy architecture, one benchmark suite.
