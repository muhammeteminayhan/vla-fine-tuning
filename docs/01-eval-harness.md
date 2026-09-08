# Phase 1 — Measurement Infrastructure

**Date:** 2026-09-08
**Gate criterion:** run the same command twice, get bit-identical results.
**Status:** ✅ passed — see §4.

We build the instrument before we build the thing it measures. Every number in
the Phase 4 report has to be traceable to a file under `results/`, and that is
only worth anything if the measurement is reproducible in the first place.

---

## 1. Why wrap `lerobot-eval` instead of calling it directly

Reading `lerobot_eval.py` (1115 lines) turned up four gaps between what it
gives us and what the experiment needs.

### 1.1 `eval_info.json` does not record per-episode seeds

`eval_policy` produces a `per_episode` list that contains the seed, but
`eval_one` drops the field when it collapses that list into `TaskMetrics`:

```python
return TaskMetrics(
    sum_rewards=[ep["sum_reward"] for ep in per_episode],
    max_rewards=[ep["max_reward"] for ep in per_episode],
    successes=[ep["success"] for ep in per_episode],   # seed is not carried over
    ...
)
```

Reading `lerobot_eval.py:528` suggests episode *i* gets `start_seed + i`. Rather
than build a schema on an unverified reading, `scripts/verify_seed_mapping.py`
spies on `eval_policy` and checks it:

```
start_seed = 1000, n_episodes = 4, batch_size = 1
task #0:  seeds = [1000, 1001, 1002, 1003]   expected = [1000, 1001, 1002, 1003]  -> MATCH
task #1:  seeds = [1000, 1001, 1002, 1003]   expected = [1000, 1001, 1002, 1003]  -> MATCH

formula 'seed = start_seed + episode_ix' holds: True
all tasks use the SAME seed sequence: True
```

The second line matters as much as the first: **every task reuses the same seed
sequence**, so a seed does not uniquely identify an episode. The schema key is
`(suite, task_id, episode_ix)`.

### 1.2 `batch_size` defaults to a machine-dependent value

```python
batch_size: int = 0   # Set to 0 for auto-tuning based on available CPU cores
```

Left at the default, results depend on how many cores the machine has. The
harness always passes it explicitly, and defaults to 1 (which also forces
`SyncVectorEnv` instead of multiprocessing).

### 1.3 Nothing upstream records provenance

No git commit, no policy path, no machine identity. Phase 4 needs all three.

### 1.4 Episode length is not recorded

`eval_main` calls `eval_policy_all(..., return_episode_data=False)`, hardcoded.
We recover episode length from the frame count of the rendered rollout video
(`ffprobe -count_frames`), verified against the 280-step cap of
`libero_object` task 0: 280 steps → 280 frames.

---

## 2. Components

| File | Role |
|---|---|
| `src/eval/run_eval.py` | The wrapper. Builds an explicit, deterministic `lerobot-eval` command, runs it, and converts `eval_info.json` into the project schema at `results/<run_id>/results.json`. |
| `src/eval/check_determinism.py` | The Phase 1 gate. Compares two runs across seven per-episode fields, video bytes, and a non-degeneracy check. |
| `src/eval/summarize.py` | Aggregates one or more runs into a table with two kinds of interval. |
| `scripts/verify_seed_mapping.py` | One-off verification of the seed formula (§1.1). |
| `scripts/run_with_gpu_peak.sh` | Runs any command while sampling GPU memory; reports peak. Phase 3 needs this on every training run. |

### `results/<run_id>/results.json` schema

```
run_id, started_at, ended_at, wall_clock_s
git      : { commit, dirty, lerobot_commit }
system   : { gpu, driver, torch, cuda, python }
policy   : { path, n_action_steps, is_base_model }
env      : { type, suite, task_ids, control_mode, init_states, hard_reset }
eval     : { n_episodes, batch_size, start_seed }
command  : [ the exact argv that produced this ]
episodes : [ { suite, task_id, episode_ix, seed, success,
               sum_reward, max_reward, episode_length, video } ]
summary  : { n_episodes, n_success, success_rate, per_task_success_rate }
failures_indexed, notes
```

Failed rollouts are symlinked into `results/<run_id>/failures/` under
self-describing names (`libero_object_task0_ep1_seed4001.mp4`) so Phase 4's
error analysis does not have to navigate lerobot's directory tree by index.

---

## 3. Two commands, not one

The K=0 reference line and the fine-tuned checkpoints **cannot be evaluated with
the same command.** `smolvla_base` declares cameras as `camera1/2/3`; LIBERO
provides `image`/`image2`. Validation
(`policies/utils.py:234`) accepts either direction being a subset, so renaming
LIBERO's two cameras onto `camera1`/`camera2` satisfies
`{camera1, camera2} ⊆ {camera1, camera2, camera3}`.

Checkpoints we fine-tune ourselves will carry LIBERO's own feature names, so
they will not need the rename. The harness handles this with `--base-model`.

---

## 4. Gate: determinism

Two runs, identical arguments (`smolvla_base`, `libero_object` tasks 0-1,
3 episodes each, seed 1000, batch_size 1):

```
  [PASS] per-episode 'task_id' identical                [0, 0, 0, 1, 1, 1] vs [0, 0, 0, 1, 1, 1]
  [PASS] per-episode 'episode_ix' identical             [0, 1, 2, 0, 1, 2] vs [0, 1, 2, 0, 1, 2]
  [PASS] per-episode 'seed' identical                   [1000, 1001, 1002, 1000, 1001, 1002] vs [...]
  [PASS] per-episode 'success' identical                [False x6] vs [False x6]
  [PASS] per-episode 'episode_length' identical         [280 x6] vs [280 x6]
  [PASS] per-episode 'sum_reward' identical             [0.0 x6] vs [0.0 x6]
  [PASS] per-episode 'max_reward' identical             [0.0 x6] vs [0.0 x6]
  [PASS] videos byte-identical across runs              6/6 match
  [PASS] rollouts differ across seeds (non-degenerate)  6 distinct hashes over 6 episodes

DETERMINISM GATE: PASSED
```

**The honest reading of this result.** CLAUDE.md asks for the `success` values
to match. They do — but `smolvla_base` fails every LIBERO episode, so that
vector is all-`False` and would match no matter how nondeterministic the system
was. On its own that check is vacuous.

What actually carries the gate is the video comparison. All six rollout videos
are byte-identical between runs, and a video encodes the entire pixel trajectory
— any nondeterminism in physics, rendering, or policy inference would change it.
The ninth check exists to close the remaining loophole: the six videos are
distinct **from each other**, so the test is not passing because the rollouts are
degenerate.

The success-vector check should be re-run in Phase 3 against a fine-tuned
checkpoint, where it will finally carry signal.

Evidence: `results/determinism_A/results.json`, `results/determinism_B/results.json`.

---

## 5. Reporting uncertainty

`summarize.py` reports two intervals, because they answer different questions.

**Wilson score interval over pooled episodes** — how precisely a given number of
binary trials pins down the rate. Wilson rather than the normal approximation
because we will spend the whole project near 0% and 100%, exactly where the
normal approximation misbehaves (it happily returns negative lower bounds).

The value of saying this out loud, from a real run:

| episodes | observed | Wilson 95% CI |
|---|---|---|
| 6 | 0/6 = 0.0% | **[0.0%, 39.0%]** |
| 18 | 0/18 = 0.0% | **[0.0%, 17.6%]** |

Zero successes out of six does not mean "0%". It means "somewhere below 39%".
Phase 4's curve is worthless without this.

**Spread across seeds** — how much the answer moves on re-run. This is the
honest error bar for Phase 4, but it needs seeds to mean anything, so it is
labelled descriptive-only below three seeds and suppressed entirely at one.

`summarize.py` refuses to aggregate runs whose policy, `n_action_steps`, suite,
control mode, or episode count differ, since averaging those would be
meaningless.

---

## 6. Measured while building this

- Eval peak VRAM: **1811 MiB of 7527 MiB.** Memory is not the constraint at eval
  time; it will be in Phase 3.
- `libero_object` task 0 episode cap: **280 steps.**
- Throughput at `n_action_steps=10`: **~10.4 s/episode**, so a full
  10-task × 10-episode suite is roughly **19 minutes per checkpoint**.

---

## 7. Gate decision

| Requirement | Status |
|---|---|
| Wrap `lerobot-eval` with fixed seeds and explicit episode counts | ✅ `src/eval/run_eval.py` |
| Schema with run_id, git hash, policy, suite, task, seed, success, length, timestamp, env | ✅ §2 |
| Summary table with per-task rate, suite average, confidence intervals | ✅ `src/eval/summarize.py` |
| Failed rollout videos saved for Phase 4 | ✅ `results/<run_id>/failures/` |
| **Determinism: same command twice → identical results** | ✅ §4 |

**Phase 1 gate: PASSED.**
