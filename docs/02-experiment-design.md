# Phase 2 — Experiment Design and Data Preparation

**Date:** 2026-09-08
**Dataset:** `lerobot/libero`, revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
**Gate criterion:** held-out split, K values, seed strategy and operator time
model written down; subset script works and is reproducible.
**Status:** ✅ passed — see §7.

Every number here comes from `results/dataset_inventory.json` and
`configs/kshot_splits.json`, both produced by scripts in `src/data/`.

---

## 1. The scenario, made concrete

| Story | Implementation |
|---|---|
| The factory's existing line | SmolVLA fine-tuned on `libero_spatial` + `libero_goal` + `libero_10` — **1239 episodes**, ~206k frames |
| A new part arrives | `libero_object` — **454 episodes** across 10 tasks, never seen during stage 1 |
| An operator teaches it | Fine-tune further on **K demonstrations per held-out task** |

Train/held-out overlap was checked, not assumed: **0 episodes in common.**

---

## 2. What is actually in the dataset

```
codebase_version : v3.0
episodes         : 1693  (metadata) / 1693 (counted)
frames           : 273465 (metadata) / 273465 (counted)
tasks            : 40    (metadata) / 40 (counted)
```

| suite | tasks | episodes | ep/task min–max | frames/episode mean | min | max |
|---|---|---|---|---|---|---|
| `libero_spatial` | 10 | 432 | 35–47 | 122.6 | 75 | 193 |
| `libero_goal` | 10 | 428 | 33–50 | 121.6 | 75 | 270 |
| `libero_object` | 10 | 454 | **42–50** | 147.5 | 114 | 254 |
| `libero_10` | 10 | 379 | 29–49 | 267.7 | 150 | 505 |

### `libero_object`, the held-out suite

| task_index | episodes | frames mean | instruction |
|---|---|---|---|
| 20 | 45 | 139.8 | pick up the orange juice and place it in the basket |
| 21 | 45 | 153.8 | pick up the ketchup and place it in the basket |
| 22 | 45 | 141.7 | pick up the cream cheese and place it in the basket |
| 23 | 46 | 146.2 | pick up the bbq sauce and place it in the basket |
| 24 | 44 | 156.1 | pick up the alphabet soup and place it in the basket |
| 25 | 45 | 143.2 | pick up the milk and place it in the basket |
| 26 | 47 | 131.9 | pick up the salad dressing and place it in the basket |
| 27 | 45 | 157.8 | pick up the butter and place it in the basket |
| 28 | **42** | 145.2 | pick up the tomato sauce and place it in the basket |
| 29 | 50 | 159.3 | pick up the chocolate pudding and place it in the basket |

**Max feasible K = 42**, set by task 28. The target `K ∈ {5, 10, 20, 40}` fits
without change.

### A mapping bug worth recording

Suite membership is resolved through LIBERO's own benchmark registry rather than
by guessing from instruction text. The first attempt keyed a dictionary on the
instruction string and iterated over all five suites, which produced a nonsense
inventory: `libero_goal` 9 tasks, `libero_10` 9, `libero_90` 2.

The cause: two instructions are shared between `libero_90` and another suite
("pick up the book and place it in the back compartment of the caddy" with
`libero_10`, "turn on the stove" with `libero_goal`). Because `libero_90` was
processed last, it silently overwrote them.

`libero_90` is not part of this dataset's 40 tasks, so it is now excluded, and
`suite_task_names()` raises on an instruction claimed by more than one suite
instead of letting iteration order decide. With that fix the inventory is a
clean 10 / 10 / 10 / 10.

---

## 3. The dataset's `fps` field is not the demonstration rate

This is the single most important correction in this phase, because the
operator time model — and therefore the x-axis of the whole project — divides by
it.

`meta/info.json` says `"fps": 10.0`, and the parquet timestamps agree: frames are
0.1 s apart, so episode 0's 214 frames appear to span 21.4 seconds.

**That is a playback label, not the rate the human worked at.** The evidence is
in LeRobot's own source, `lerobot/envs/libero.py:98`:

```python
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,   # longest training demo has 254 steps
}
```

Our measured maxima are **193 frames** for `libero_spatial` and **254 frames**
for `libero_object` — both exact matches. So one dataset frame is one
environment control step, and LIBERO's robosuite control loop runs at
**20 Hz**.

| | mean demo duration |
|---|---|
| Using the metadata's `fps = 10` | 14.75 s |
| Using the true control rate, 20 Hz | **7.37 s** |

Taking the metadata at face value would have doubled every operator-minute
figure in the final report. `CONTROL_HZ = 20.0` is defined once, in
`src/data/make_kshot_split.py`, with this reasoning in a comment beside it.

---

## 4. Operator time model

```
operator_seconds_per_demo = demo_seconds + reset_seconds
demo_seconds              = mean_frames / 20 Hz = 147.5 / 20 = 7.37 s   [measured]
reset_seconds             = 20 s                                        [ASSUMED]
minutes_per_task(K)       = K x (7.37 + 20) / 60
```

### The assumption, stated plainly

The recorded trajectory covers only the motion that was kept. It does **not**
include putting the object back to a fresh start pose, returning the arm to
home, the operator re-gripping the teleoperation device, starting and stopping
the recording, or attempts that were discarded. For a tabletop pick-and-place
whose recorded motion is about seven seconds, the surrounding overhead is
plausibly two to five times the motion itself. **20 s is a judgement call, not a
measurement**, and this project has no way to measure it — the demonstrations
were collected by someone else, and no wall-clock timing survives in the data.

### Why this matters more than it looks

At `reset = 20 s`, the assumed part is **73% of the total operator time**. The
x-axis of the headline graph is therefore mostly an assumption resting on a
small measurement. Hiding that would make the business argument look far more
solid than it is.

Sensitivity, in minutes of operator time per new task:

| K | reset = 10 s | reset = 20 s | reset = 40 s |
|---|---|---|---|
| 5 | 1.4 | **2.3** | 4.0 |
| 10 | 2.9 | **4.6** | 7.9 |
| 20 | 5.8 | **9.1** | 15.8 |
| 40 | 11.6 | **18.2** | 31.6 |

Because operator minutes are just `K x constant`, changing the reset assumption
rescales the x-axis but never changes the shape of the curve or where it
saturates. Phase 4 must therefore plot the curve against K and label operator
minutes as a derived axis, showing this range rather than a single number.

---

## 5. K-shot subsets

`src/data/make_kshot_split.py` → `configs/kshot_splits.json`.

**K values:** 5, 10, 20, 40. **Seeds:** 0, 1, 2.

Two properties, both verified rather than asserted:

**Reproducible.** Each task draws from its own RNG stream seeded by
`(seed, task_index)`, so adding or reordering tasks cannot disturb another
task's draw. Running the generator twice produced byte-identical output
(same sha256). Different seeds produce genuinely different subsets — `k5_seed0`
and `k5_seed1` share only 6 of 50 episodes.

**Nested.** `K=5 ⊂ K=10 ⊂ K=20 ⊂ K=40`. Each task's episodes are shuffled once
and every K takes a prefix; the script asserts the subset relation for every
seed and every adjacent pair of K values before writing anything. For task 24,
seed 0, all four K values start `[1224, 1143, 1171, 887, 1187]`.

Without nesting, going from K=5 to K=10 would change *which* demonstrations the
model sees as well as *how many*, and the curve would confound "more data" with
"different data".

| K | episodes (10 tasks) | min/task | min for all 10 |
|---|---|---|---|
| 5 | 50 | 2.3 | 22.8 |
| 10 | 100 | 4.6 | 45.6 |
| 20 | 200 | 9.1 | 91.2 |
| 40 | 400 | 18.2 | 182.5 |

Each split carries its own `--dataset.episodes='[...]'` argument, so Phase 3
consumes the split file directly rather than re-deriving indices.

---

## 6. Open risk: the compute budget has not been measured

The design above implies **13 training runs**: one stage-1 run on 1239 episodes,
then 4 K values x 3 seeds.

The LeRobot SmolVLA documentation states that 20k steps takes roughly 4 hours on
a single A100. This project has a laptop RTX 5060 with 7.36 GiB, which will also
force a much smaller batch size. **We have not measured our own throughput**, so
the total cost of Phase 3 is currently unknown, and it could be large enough to
require changing the design.

This is flagged here rather than discovered mid-Phase-3. The first action of
Phase 3 should be a throughput and batch-size measurement; if the budget does
not fit, the design must be revised **before** any run starts, with the change
agreed explicitly (CLAUDE.md rule 6).

---

## 7. Gate decision

| Requirement | Status |
|---|---|
| Held-out split defined, leakage checked | ✅ §1 — 0 episodes shared |
| Dataset inspected, real numbers written down | ✅ §2 |
| K values chosen against real per-task counts | ✅ §2 — max feasible 42, so {5,10,20,40} stands |
| Seed strategy | ✅ §5 — seeds 0/1/2, per-task RNG streams |
| Operator time model with stated, justified assumption | ✅ §4 — including its sensitivity |
| Subset script works and is reproducible | ✅ §5 — identical sha256 across runs, nesting asserted |

**Phase 2 gate: PASSED**, with the Phase 3 compute risk in §6 recorded as open.
