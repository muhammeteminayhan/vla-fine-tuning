#!/usr/bin/env python3
"""Phase 4: which held-out tasks are genuinely hard, pooled over every run.

Per-K, per-seed task rates rest on 10-15 episodes, where the Wilson interval is
roughly 50 points wide - too coarse to rank anything. Pooling every run for a
task (all K, all seeds) gives ~45 episodes and intervals narrow enough to say
something. The cost is that the pooled number mixes K values, so it answers
"which tasks are hard for this method" rather than "how does task difficulty
change with K".
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="results/stage2_k*_seed*")
    ap.add_argument("--json-out", default="results/task_difficulty.json")
    a = ap.parse_args()

    inv = json.loads((REPO / "results/dataset_inventory.json").read_text())
    per_task = inv["libero_object_per_task"]
    names = {i: per_task[t]["task"] for i, t in enumerate(sorted(per_task, key=int))}

    pooled: dict[int, list[bool]] = defaultdict(list)
    runs = 0
    for f in sorted(REPO.glob(a.pattern)):
        r = f / "results.json"
        if not r.exists():
            continue
        runs += 1
        for e in json.loads(r.read_text())["episodes"]:
            pooled[e["task_id"]].append(e["success"])

    rows = []
    for t in sorted(pooled):
        v = pooled[t]
        lo, hi = wilson(sum(v), len(v))
        rows.append({"task_id": t, "task": names.get(t), "n": len(v),
                     "n_success": sum(v), "rate": sum(v) / len(v),
                     "wilson_lo": lo, "wilson_hi": hi})
    rows.sort(key=lambda r: -r["rate"])

    print(f"pooled over {runs} runs, {rows[0]['n']} episodes per task\n")
    print(f"{'task':>4} {'success':>12} {'Wilson 95%':>18}  instruction")
    for r in rows:
        print(f"{r['task_id']:>4} {r['n_success']:>3}/{r['n']:<3} {r['rate']*100:>4.0f}% "
              f"[{r['wilson_lo']*100:>5.1f}%,{r['wilson_hi']*100:>6.1f}%]  {r['task']}")

    best, worst = rows[0], rows[-1]
    separated = worst["wilson_hi"] < best["wilson_lo"]
    print(f"\nhardest vs easiest: {worst['rate']*100:.0f}% vs {best['rate']*100:.0f}% "
          f"({(best['rate']-worst['rate'])*100:.0f} points)")
    print(f"intervals disjoint: {separated}")

    (REPO / a.json_out).write_text(json.dumps(
        {"n_runs": runs, "tasks": rows,
         "hardest": worst["task_id"], "easiest": best["task_id"],
         "intervals_disjoint": separated}, indent=2) + "\n")
    print(f"\nwrote {REPO / a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
