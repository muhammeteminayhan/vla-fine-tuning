#!/usr/bin/env python3
"""Phase 1 gate: prove that two identical eval runs produce identical results.

CLAUDE.md asks for the `success` values to match. That is necessary but, on its
own, weak: before any fine-tuning every episode fails, so the success vector is
all-False and matches trivially. A gate that passes on an all-zeros vector
proves nothing.

So this checks four things, from weakest to strongest:

  1. success vector            - what CLAUDE.md asks for
  2. episode lengths, rewards  - richer, still coarse
  3. video bytes are identical - the strong one: the video encodes the entire
                                 pixel trajectory, so any nondeterminism in
                                 physics, rendering or policy would show up
  4. videos DIFFER across seeds - guards against the test passing because the
                                 rollouts are degenerate rather than reproducible
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def load(run_dir: Path) -> dict:
    return json.loads((run_dir / "results.json").read_text())


def video_hash(run_dir: Path, episode: dict) -> str | None:
    if not episode.get("video"):
        return None
    path = run_dir / "lerobot_raw" / episode["video"]
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    a_args = p.parse_args()

    A, B = load(a_args.run_a), load(a_args.run_b)
    checks: list[tuple[str, bool, str]] = []

    # 1-2. per-episode fields
    for key in ("task_id", "episode_ix", "seed", "success", "episode_length",
                "sum_reward", "max_reward"):
        va = [e[key] for e in A["episodes"]]
        vb = [e[key] for e in B["episodes"]]
        checks.append((f"per-episode '{key}' identical", va == vb, f"{va} vs {vb}"))

    # 3. video bytes
    ha = [video_hash(a_args.run_a, e) for e in A["episodes"]]
    hb = [video_hash(a_args.run_b, e) for e in B["episodes"]]
    have_videos = all(h is not None for h in ha + hb) and len(ha) > 0
    checks.append(("videos byte-identical across runs", have_videos and ha == hb,
                   f"{sum(x == y for x, y in zip(ha, hb))}/{len(ha)} match"))

    # 4. non-degeneracy: different seeds must give different rollouts
    distinct = len(set(h for h in ha if h))
    checks.append(("rollouts differ across seeds (non-degenerate)",
                   distinct == len(ha) and len(ha) > 1,
                   f"{distinct} distinct hashes over {len(ha)} episodes"))

    # Report
    width = max(len(name) for name, _, _ in checks)
    print(f"A = {a_args.run_a}\nB = {a_args.run_b}\n")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    passed = all(ok for _, ok, _ in checks)
    n_success = A["summary"]["n_success"]
    print(f"\nDETERMINISM GATE: {'PASSED' if passed else 'FAILED'}")
    if passed and n_success == 0:
        print(
            "\nNOTE: every episode failed (n_success=0), so the success-vector check\n"
            "      is vacuous on its own. This gate passes on the video-hash evidence.\n"
            "      Re-run it against a fine-tuned checkpoint in Phase 3, where the\n"
            "      success vector carries real signal."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
