#!/usr/bin/env python3
"""Phase 1: aggregate one or more eval runs into a table with uncertainty.

Two different intervals are reported, because they answer different questions
and quoting only one of them would overstate what we know:

  * Wilson interval over pooled episodes - "given this many binary trials, how
    precisely do we know the success rate?" Wilson rather than the normal
    approximation because rates near 0 or 1 are exactly where we will live, and
    the normal approximation is badly wrong there (it can produce negative
    lower bounds).

  * Spread across seeds - "how much does the answer move when we re-run with a
    different seed?" This is the honest one for Phase 4's error bars, but it
    needs several seeds to mean anything, so it is only reported when there are
    at least two and is labelled as descriptive when there are fewer than three.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Well behaved at p near 0 and 1, unlike normal approx."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for p in paths:
        f = p / "results.json" if p.is_dir() else p
        runs.append(json.loads(f.read_text()))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path, nargs="+", help="run directories or results.json files")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    runs = load_runs(a.runs)

    # Guard against silently averaging incomparable runs.
    configs = {(r["policy"]["path"], r["policy"]["n_action_steps"], r["env"]["suite"],
                r["env"]["control_mode"], r["eval"]["n_episodes"]) for r in runs}
    if len(configs) > 1:
        print("REFUSING TO AGGREGATE: runs differ in configuration, so an average "
              "would not mean anything.")
        for c in sorted(map(str, configs)):
            print(f"  {c}")
        return 1

    per_task = defaultdict(list)          # task key -> successes across all runs
    per_seed_rate: dict[int, list] = defaultdict(list)
    for r in runs:
        start_seed = r["eval"]["start_seed"]
        for e in r["episodes"]:
            per_task[f"{e['suite']}/{e['task_id']}"].append(e["success"])
            per_seed_rate[start_seed].append(e["success"])

    all_success = [s for v in per_task.values() for s in v]
    n, k = len(all_success), sum(all_success)
    lo, hi = wilson(k, n)

    seeds = sorted(per_seed_rate)
    seed_rates = [sum(v) / len(v) for _, v in sorted(per_seed_rate.items())]

    print(f"policy      : {runs[0]['policy']['path']}  (n_action_steps={runs[0]['policy']['n_action_steps']})")
    print(f"suite       : {runs[0]['env']['suite']}   control_mode={runs[0]['env']['control_mode']}")
    print(f"runs        : {len(runs)}   seeds={seeds}   episodes={n}\n")

    print(f"| {'task':<22} | {'n':>4} | {'success':>7} | {'rate':>7} |")
    print(f"|{'-'*24}|{'-'*6}|{'-'*9}|{'-'*9}|")
    for task in sorted(per_task):
        v = per_task[task]
        print(f"| {task:<22} | {len(v):>4} | {sum(v):>7} | {sum(v)/len(v)*100:>6.1f}% |")
    print(f"|{'-'*24}|{'-'*6}|{'-'*9}|{'-'*9}|")
    print(f"| {'SUITE TOTAL':<22} | {n:>4} | {k:>7} | {k/n*100:>6.1f}% |")

    print(f"\nWilson 95% CI over {n} pooled episodes : [{lo*100:.1f}%, {hi*100:.1f}%]")

    if len(seeds) >= 2:
        mean = sum(seed_rates) / len(seed_rates)
        var = sum((x - mean) ** 2 for x in seed_rates) / (len(seed_rates) - 1)
        sd = math.sqrt(var)
        label = "95% CI" if len(seeds) >= 3 else "descriptive only (n=2)"
        half = 1.96 * sd / math.sqrt(len(seed_rates)) if len(seeds) >= 3 else sd
        print(f"across {len(seeds)} seeds: mean={mean*100:.1f}%  sd={sd*100:.1f}pp  "
              f"min={min(seed_rates)*100:.1f}%  max={max(seed_rates)*100:.1f}%")
        print(f"  seed-level {label}: {(mean-half)*100:.1f}% .. {(mean+half)*100:.1f}%")
    else:
        print(f"across seeds: only {len(seeds)} seed present - no across-seed interval. "
              "Phase 4 needs at least 3.")

    if a.json_out:
        payload = {
            "policy": runs[0]["policy"], "env": runs[0]["env"],
            "n_runs": len(runs), "seeds": seeds, "n_episodes": n, "n_success": k,
            "success_rate": k / n,
            "wilson95": [lo, hi],
            "per_task": {t: {"n": len(v), "k": sum(v), "rate": sum(v) / len(v)}
                         for t, v in sorted(per_task.items())},
            "per_seed_rate": dict(zip(seeds, seed_rates)),
            "source_runs": [r["run_id"] for r in runs],
        }
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
