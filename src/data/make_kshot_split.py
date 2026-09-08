#!/usr/bin/env python3
"""Phase 2: build reproducible K-shot subsets of the held-out suite.

Two properties matter more than anything else here.

**Reproducibility.** The same seed must always produce the same subset, on any
machine, forever. Each task gets its own RNG stream seeded from (seed,
task_index), so adding or reordering tasks cannot disturb another task's draw.

**Nesting.** The K=5 subset is a strict subset of K=10, which is a strict subset
of K=20, and so on. Each task's episodes are shuffled once and every K takes a
prefix. Without nesting, moving from K=5 to K=10 would change *which*
demonstrations the model sees as well as *how many*, and the curve would no
longer isolate the effect we are trying to measure.

Operator time is derived from the true control rate, not the dataset's fps
metadata field - see docs/02-experiment-design.md for why those differ.
"""

import argparse
import json
from pathlib import Path

import numpy as np

# One dataset frame is one environment control step, and LIBERO's robosuite
# control loop runs at 20 Hz. The dataset metadata says fps=10.0, which is a
# playback label, not the rate the demonstration was performed at. Confirmed by
# lerobot/envs/libero.py, whose TASK_SUITE_MAX_STEPS comments record the longest
# training demo as 193 steps (libero_spatial) and 254 steps (libero_object) -
# exactly the frame counts in this dataset.
CONTROL_HZ = 20.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", default="results/dataset_inventory.json")
    ap.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--reset-seconds", type=float, default=20.0,
                    help="operator time per demonstration outside the recorded motion")
    ap.add_argument("--out", default="configs/kshot_splits.json")
    a = ap.parse_args()

    inv = json.loads(Path(a.inventory).read_text())
    per_task = inv["libero_object_per_task"]
    task_ids = sorted(per_task, key=int)

    available = {t: len(per_task[t]["episode_indices"]) for t in task_ids}
    max_k = min(available.values())
    too_big = [k for k in a.k_values if k > max_k]
    if too_big:
        raise SystemExit(
            f"K values {too_big} exceed the smallest task's episode count ({max_k}). "
            f"Per-task availability: {available}"
        )

    # Per-task mean demo duration, in seconds of real operator effort.
    mean_frames = {t: per_task[t]["frames_mean"] for t in task_ids}
    suite_mean_frames = float(np.mean(list(mean_frames.values())))
    suite_mean_demo_s = suite_mean_frames / CONTROL_HZ

    splits = {}
    for seed in a.seeds:
        # Shuffle each task once; every K is a prefix, so the subsets nest.
        order = {}
        for t in task_ids:
            eps = np.array(per_task[t]["episode_indices"])
            rng = np.random.default_rng([seed, int(t)])
            order[t] = eps[rng.permutation(len(eps))].tolist()

        for k in a.k_values:
            chosen = {t: order[t][:k] for t in task_ids}
            flat = sorted(e for v in chosen.values() for e in v)
            demo_s = sum(mean_frames[t] for t in task_ids) / len(task_ids) / CONTROL_HZ
            per_task_s = k * (demo_s + a.reset_seconds)
            splits[f"k{k}_seed{seed}"] = {
                "K": k,
                "seed": seed,
                "n_tasks": len(task_ids),
                "n_episodes_total": len(flat),
                "episodes_per_task": {t: chosen[t] for t in task_ids},
                "episodes": flat,
                "operator_time": {
                    "control_hz": CONTROL_HZ,
                    "reset_seconds_assumed": a.reset_seconds,
                    "mean_demo_seconds": round(demo_s, 2),
                    "seconds_per_demo_total": round(demo_s + a.reset_seconds, 2),
                    "minutes_per_task": round(per_task_s / 60, 2),
                    "minutes_all_10_tasks": round(per_task_s * len(task_ids) / 60, 2),
                },
                "lerobot_train_arg": f"--dataset.episodes='{json.dumps(flat)}'",
            }

    # Verify nesting rather than trusting the construction.
    for seed in a.seeds:
        ks = sorted(a.k_values)
        for small, large in zip(ks, ks[1:]):
            s = set(splits[f"k{small}_seed{seed}"]["episodes"])
            l = set(splits[f"k{large}_seed{seed}"]["episodes"])
            if not s.issubset(l):
                raise SystemExit(f"nesting broken: k{small} not a subset of k{large} (seed {seed})")

    payload = {
        "dataset_revision": inv["revision"],
        "held_out_suite": "libero_object",
        "training_suites": ["libero_spatial", "libero_goal", "libero_10"],
        "k_values": a.k_values,
        "seeds": a.seeds,
        "max_feasible_k": max_k,
        "episodes_available_per_task": available,
        "control_hz": CONTROL_HZ,
        "reset_seconds_assumed": a.reset_seconds,
        "suite_mean_demo_seconds": round(suite_mean_demo_s, 2),
        "nesting_verified": True,
        "splits": splits,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"held-out suite : libero_object ({len(task_ids)} tasks)")
    print(f"episodes/task  : {min(available.values())}-{max(available.values())}  -> max feasible K = {max_k}")
    print(f"mean demo      : {suite_mean_frames:.1f} frames / {CONTROL_HZ:.0f} Hz = {suite_mean_demo_s:.2f} s")
    print(f"reset assumed  : {a.reset_seconds:.0f} s per demonstration\n")
    print(f"| {'K':>3} | {'episodes':>8} | {'min/task':>9} | {'min all 10':>11} |")
    print(f"|{'-'*5}|{'-'*10}|{'-'*11}|{'-'*13}|")
    for k in a.k_values:
        s = splits[f"k{k}_seed{a.seeds[0]}"]
        ot = s["operator_time"]
        print(f"| {k:>3} | {s['n_episodes_total']:>8} | {ot['minutes_per_task']:>9.1f} | "
              f"{ot['minutes_all_10_tasks']:>11.1f} |")
    print(f"\nnesting verified across {len(a.seeds)} seeds")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
