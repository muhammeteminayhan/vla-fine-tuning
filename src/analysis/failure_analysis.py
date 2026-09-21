#!/usr/bin/env python3
"""Phase 4: what the failures have in common, from the data rather than from
watching.

Two things can be said rigorously from the run records, before any video is
opened:

  * whether failures are near-misses on time, or not close at all
  * how failures distribute over tasks and over K

Classifying *why* each rollout failed (wrong object, missed grasp, early
release, missed target) needs more than the agentview video this harness keeps;
see the limitation noted at the bottom of the output.
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="results/stage2_k*_seed*")
    ap.add_argument("--json-out", default="results/failure_analysis.json")
    a = ap.parse_args()

    succ, fail = [], []
    resolutions: set[int] = set()
    by_task = defaultdict(lambda: {"n": 0, "fail": 0})
    by_k = defaultdict(lambda: {"n": 0, "fail": 0})
    for f in sorted(REPO.glob(a.pattern)):
        r = f / "results.json"
        if not r.exists():
            continue
        K = int(f.name.split("_")[1][1:])
        d = json.loads(r.read_text())
        resolutions.add(d["eval"]["n_episodes"])
        for e in d["episodes"]:
            (succ if e["success"] else fail).append(e.get("episode_length"))
            by_task[e["task_id"]]["n"] += 1
            by_k[K]["n"] += 1
            if not e["success"]:
                by_task[e["task_id"]]["fail"] += 1
                by_k[K]["fail"] += 1

    if len(resolutions) > 1:
        raise SystemExit(
            f"refusing to pool runs evaluated at different resolutions: "
            f"{sorted(resolutions)} episodes per task matched {a.pattern!r}."
        )

    succ = [x for x in succ if x is not None]
    fail = [x for x in fail if x is not None]
    cap = max(fail + succ)
    at_cap = sum(1 for x in fail if x >= cap)

    print(f"successes : n={len(succ):>4}  mean={st.mean(succ):>6.1f} steps  "
          f"median={st.median(succ):>5.0f}  min={min(succ)}  max={max(succ)}")
    print(f"failures  : n={len(fail):>4}  mean={st.mean(fail):>6.1f} steps  "
          f"all at the {cap}-step cap: {at_cap}/{len(fail)}")
    print()
    print("The environment terminates early only on success (LiberoEnv.step:")
    print("`terminated = done or is_success`), so failures reaching the cap is")
    print("expected. What is informative is the ratio:")
    print(f"  a typical success needs {st.median(succ):.0f} steps, the cap is {cap}")
    print(f"  -> failures had ~{cap / st.median(succ):.1f}x the time a success needs")
    print("  -> these are not near-misses on time; a larger step budget would not")
    print("     convert them into successes.")

    print("\nfailures per K:")
    for K in sorted(by_k):
        d = by_k[K]
        print(f"  K={K:<3} {d['fail']:>3}/{d['n']:<4} = {d['fail']/d['n']*100:>4.1f}%")

    print("\nfailures per task:")
    for t in sorted(by_task, key=lambda x: -by_task[x]["fail"]):
        d = by_task[t]
        print(f"  task {t:>2}  {d['fail']:>3}/{d['n']:<4} = {d['fail']/d['n']*100:>4.1f}%")

    payload = {
        "success_steps": {"n": len(succ), "mean": st.mean(succ),
                          "median": st.median(succ), "min": min(succ), "max": max(succ)},
        "failure_steps": {"n": len(fail), "mean": st.mean(fail),
                          "cap": cap, "at_cap": at_cap},
        "time_headroom_ratio": cap / st.median(succ),
        "per_k": {str(k): by_k[k] for k in sorted(by_k)},
        "per_task": {str(t): by_task[t] for t in sorted(by_task)},
        "limitation": (
            "Failure *modes* are not classified here. The eval video is the "
            "agentview only (LiberoEnv.render returns the first camera), at "
            "360x360, with no gripper state or object poses recorded. "
            "Separating 'wrong object' from 'missed grasp' reliably would need "
            "the harness to log object poses or the wrist camera."
        ),
    }
    (REPO / a.json_out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {REPO / a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
