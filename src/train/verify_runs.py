#!/usr/bin/env python3
"""Phase 3 gate: prove no run silently differs from the others.

CLAUDE.md's gate for this phase is that no run changed configuration behind our
backs. Eyeballing a dozen run records does not prove that. This compares every
run_record.json field by field and fails if anything varies that is not supposed
to - K, the seed, and the things that follow from them.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Fields expected to differ between runs, and only these.
EXPECTED_TO_VARY = {
    "split", "K", "seed", "n_episodes", "started_at", "wall_clock_s",
    "wall_clock_h", "gpu_peak_mib", "final_loss", "loss_first_logged",
    "median_step_s", "checkpoints", "log", "command", "git_commit", "init_from",
}


def main() -> int:
    records = {}
    for f in sorted(REPO.glob("checkpoints/stage2/*/run_record.json")):
        records[f.parent.name] = json.loads(f.read_text())
    stage1 = REPO / "checkpoints/stage1/run_record.json"

    if not records:
        print("no stage-2 run records found")
        return 1

    print(f"comparing {len(records)} stage-2 runs: {', '.join(sorted(records))}\n")

    values = defaultdict(set)
    for name, rec in records.items():
        for k, v in rec.items():
            values[k].add(json.dumps(v, sort_keys=True))

    drift = {k: v for k, v in values.items() if len(v) > 1 and k not in EXPECTED_TO_VARY}
    fixed = {k: next(iter(v)) for k, v in values.items()
             if len(v) == 1 and k not in EXPECTED_TO_VARY}

    print("identical across every run:")
    for k in sorted(fixed):
        print(f"  {k:<20} = {fixed[k]}")

    if drift:
        print("\nUNEXPECTED VARIATION:")
        for k, v in sorted(drift.items()):
            print(f"  {k}: {sorted(v)}")
    else:
        print("\nno unexpected variation")

    # Every stage-2 run must start from the same stage-1 checkpoint.
    inits = {r["init_from"] for r in records.values()}
    init_ok = len(inits) == 1
    print(f"\nall runs initialised from one checkpoint: {init_ok}  {sorted(inits)}")

    # And every run must have finished cleanly.
    bad = [n for n, r in records.items() if not r["ok"]]
    print(f"runs that did not complete cleanly: {bad or 'none'}")

    if stage1.exists():
        s = json.loads(stage1.read_text())
        print(f"\nstage 1: {s['n_episodes']} episodes, {s['steps']} steps, "
              f"loss {s['loss_first_logged']} -> {s['final_loss']}, ok={s['ok']}")

    ok = not drift and init_ok and not bad
    print(f"\nCONFIG CONSISTENCY: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
