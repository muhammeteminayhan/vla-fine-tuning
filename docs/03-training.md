# Phase 3 — Training Loop

**Date:** 2026-09-09
**Gate criterion:** checkpoints for all K values, complete training logs, no
silent configuration change in any run.
**Status:** in progress — measurement and decisions below are final; run results
are filled in as the driver completes them.

Sources: `results/batch_size_sweep/results.json`, `results/compute_budget.json`,
`checkpoints/**/run_record.json`.

---

## 1. What fits in 7.36 GiB, measured

CLAUDE.md forbids guessing a batch size and forbids silently lowering one after
an OOM. The sweep therefore treats the OOM as the deliberate end of the search
and stops there.

Fixed across the sweep: `smolvla_base`, LoRA r=64 α=64, bf16, the K=40 seed-0
split, 60 steps per point.

| batch_size | peak VRAM | % of 7527 MiB | step_s | samples/s |
|---|---|---|---|---|
| 1 | 1327 MiB | 18% | 0.091 | 10.99 |
| 2 | 1515 MiB | 20% | 0.112 | 17.86 |
| 4 | 1895 MiB | 25% | 0.172 | **23.26** |
| 8 | 2623 MiB | 35% | 0.340 | 23.53 |
| 16 | 4075 MiB | 54% | 0.706 | 22.66 |
| 32 | 7071 MiB | 94% | 1.438 | 22.25 |
| 64 | **OOM** at 7679 MiB | — | — | — |

**Throughput saturates at batch_size 4.** Going from 4 to 32 buys nothing in
samples/s while quadrupling memory. The familiar "bigger batch is faster" rule
only holds while the GPU is still under-fed; past that point a larger batch
just costs memory.

`batch_size=16` was chosen for the study: it sits at 54% of the card, leaving
real headroom, while staying close to the 32 used in LeRobot's PEFT example so
that gradient-noise behaviour is not wildly different from the documented setup.
32 fits, but at 94% of VRAM any fluctuation would OOM.

### Why LoRA makes this possible at all

```
num_learnable_params =   2,970,624  (3M)
num_total_params     = 453,016,800  (453M)      -> 0.66%
```

The saving is not mainly in the weights, it is in the optimizer state. Adam
keeps two extra tensors per trainable parameter. For 453M parameters in fp32
that is roughly 3.6 GB; for 3M it is about 24 MB. That difference is what lets a
453M-parameter model fine-tune on a 7.36 GiB laptop GPU.

Each saved checkpoint is **11.9 MB** — the adapter only, not the 865 MB base.

---

## 2. Wall-clock budget, derived from the measurement

Plateau throughput: **22.93 samples/s**.

| steps per run | h / run | 12 runs | + stage 1 | total |
|---|---|---|---|---|
| 100,000 (docs default) | 19.4 | 232.6 | 38.8 | **271 h** |
| 20,000 | 3.9 | 46.5 | 7.8 | 54 h |
| 10,000 | 1.9 | 23.3 | 3.9 | 27 h |
| 6,000 | 1.2 | 14.0 | 3.5 | **17.5 h** |

The LeRobot documentation's default of 100k steps would take eleven days on this
machine. This is the answer to the open risk recorded in
`docs/02-experiment-design.md §6`.

---

## 3. Step count, decided by measurement

Two pilots on the same split (K=40, seed 0), same eval protocol, comparing
**final** checkpoints:

| steps | training | success | Wilson 95% |
|---|---|---|---|
| 2000 | 0.40 h | 24/50 = 48% | [34.8%, 61.5%] |
| 6000 | 1.18 h | 36/50 = **72%** | [58.3%, 82.5%] |

Three times the compute for +24 points, with intervals that barely overlap.
**Fixed at 6000 steps for every run in the study.**

### Why the comparison had to be a separate run

LeRobot auto-scales the LR schedule to the step budget:

```
Auto-scaling LR scheduler: num_training_steps (2000) < num_decay_steps (30000).
Scaling warmup: 1000 -> 66, decay: 30000 -> 2000 (scale factor: 0.067)
```

Each run compresses a full warmup-and-decay cycle into whatever step count it is
given. Two consequences:

1. Intermediate checkpoints are points on *that run's* schedule, not a universal
   training curve. Step 1500 of a 2000-step run and step 1500 of a 6000-step run
   sit at completely different learning rates.
2. Runs with different total step counts are not comparable except at their
   final checkpoints — which is why the step count must be identical across all
   K. That is a technical requirement, not only a fairness one.

It also makes the **final checkpoint the natural pre-registered choice**, which
removes the temptation to report whichever checkpoint scored best.

### The 2000-step run was not monotonic

| steps | success | Wilson 95% |
|---|---|---|
| 500 | 7/50 = 14% | [7.0%, 26.2%] |
| 1000 | 11/50 = 22% | [12.8%, 35.2%] |
| 1500 | 35/50 = 70% | [56.2%, 80.9%] |
| 2000 | 24/50 = 48% | [34.8%, 61.5%] |

Step 1500 scored well above the final step. The tempting explanation is
overfitting, and it is wrong: 2000 steps × 16 = 32,000 samples against a
59,017-frame dataset is **0.54 epochs**, so the model had not seen the data even
once. Checking that number before writing the explanation is the only reason a
wrong one did not end up in this document. The likelier causes are the
compressed LR schedule and eval noise at 50 episodes (±13 points).

---

## 4. Two-stage design and how the adapter is carried over

- **Stage 1 — the factory's existing line.** `smolvla_base` + LoRA on
  `libero_spatial` + `libero_goal` + `libero_10` (1239 episodes).
- **Stage 2 — teaching a new part.** Continue from the stage-1 checkpoint on K
  demonstrations per `libero_object` task.

Initialising from a PEFT checkpoint does **not** nest a second adapter.
LeRobot detects the existing one:

```
PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.
num_learnable_params=2970624 (3M)
```

The second line is the one that mattered. PEFT adapters are often loaded frozen,
and had that happened here every stage-2 run would have trained nothing and
silently returned its stage-1 input — producing a perfectly flat, entirely
meaningless curve. Verified by running 20 steps from a real checkpoint and
reading the count, rather than by reasoning about it.

---

## 5. Configuration held fixed across every run

Defined as constants in `src/train/run_train.py` rather than as CLI defaults, so
they cannot drift between runs, and written into each `run_record.json`.

| | value |
|---|---|
| LoRA | `method_type=LORA`, `r=64`, `lora_alpha=64` (scaling = 1.0) |
| target modules | LeRobot default: `q_proj`/`v_proj` of the LM expert + state/action projections |
| optimizer | `optimizer_lr=1e-3`, `scheduler_decay_lr=1e-4` (10x the full fine-tune rate, per the PEFT docs) |
| precision | bf16 |
| batch size | 16 |
| steps | 6000 |
| `n_action_steps` | 10 (decided in Phase 1) |
| dataset revision | `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |

Only K and the seed change.

---

## 6. Run results

Every run below used the configuration in §5; `src/train/verify_runs.py` checks
that mechanically rather than by eye and currently reports **CONFIG CONSISTENCY:
PASS** — identical steps, batch size, LoRA config, optimizer, precision,
`n_action_steps` and dataset revision across all of them, all initialised from
the same stage-1 checkpoint.

| run | K | seed | episodes | wall | peak VRAM | loss | success |
|---|---|---|---|---|---|---|---|
| `stage1` | — | 0 | 1239 | 1.20 h | 4075 MiB | 2.435 → 0.495 | 0/50 = 0% (held-out) |
| `k5_seed0` | 5 | 0 | 50 | 1.18 h | 4149 MiB | 0.559 → 0.094 | 29/50 = 58% |
| `k10_seed0` | 10 | 0 | 100 | 1.18 h | 4127 MiB | 0.564 → 0.169 | 36/50 = 72% |
| `k20_seed0` | 20 | 0 | 200 | 1.18 h | 4145 MiB | 0.559 → 0.254 | 32/50 = 64% |
| `k40_seed0` | 40 | 0 | 400 | 1.18 h | 4149 MiB | 0.558 → 0.305 | 41/50 = 82% |
| `k5_seed1` | 5 | 1 | 50 | 1.18 h | 4127 MiB | 0.568 → 0.102 | 34/50 = 68% |
| `k10_seed1` | 10 | 1 | 100 | 1.18 h | 4153 MiB | 0.573 → 0.173 | 34/50 = 68% |
| `k20_seed1` | 20 | 1 | 200 | 1.18 h | 4123 MiB | 0.574 → 0.272 | 43/50 = 86% |
| `k40_seed1` | 40 | 1 | 400 | 1.18 h | 4123 MiB | 0.588 → 0.298 | 41/50 = 82% |
| `k5_seed2` | 5 | 2 | 50 | 1.18 h | 4127 MiB | 0.556 → 0.098 | (pending) |

Total training so far: **11.8 h** on one RTX 5060 Laptop. Peak VRAM never
exceeded 4153 MiB
of 7527, so batch_size 16 left the headroom it was chosen for.

### Training loss runs the wrong way

Final loss falls as K falls — 0.094 at K=5 against 0.305 at K=40 — while success
rate moves the other way. With the step count fixed, a smaller K means more
epochs over less data, so the model fits its handful of demonstrations more and
more tightly. It gets very good at reproducing them and less good at the task.

Loss is therefore not a usable model-selection signal in this study, and a run
that "converged better" is not a better policy. Everything reported here is
measured by rollout success in the environment.

Success rates above are per run at 5 episodes per task; the pooled per-K figures
and their intervals live in `results/teaching_cost_curve.json`.

---

## 7. Known gaps

- **Catastrophic forgetting is not measured.** Stage 2 continues training the
  stage-1 adapter, so teaching a new part may degrade the tasks the line already
  knew. Measuring it is cheap — evaluate a stage-2 checkpoint on
  `libero_spatial` — and it belongs in the limitations section either way.
- **Eval noise.** Overnight runs use 5 episodes per task (50 total), giving a
  Wilson width of roughly ±13 points. The published LIBERO protocol is 10 per
  task; Phase 4 should re-run at that resolution before any number is reported
  as final.
