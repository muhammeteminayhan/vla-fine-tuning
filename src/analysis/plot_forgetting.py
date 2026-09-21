#!/usr/bin/env python3
"""Phase 4: what teaching a new part costs the old line.

The teaching cost curve answers "can an operator teach a new part cheaply". It
cannot show the price: the same checkpoints that score well on the new part
score zero on the suite the model was trained on. That belongs in its own
figure rather than as a footnote, because it changes what the headline means.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.stats import wilson  # noqa: E402

BARS = [
    ("stage 1\n(before teaching)", "forgetting_stage1_spatial", "stage1_baseline"),
    ("+ 5 demos\nof a new part", "forgetting_k5_seed0_spatial", "stage2_k5_seed0"),
    ("+ 40 demos\nof a new part", "forgetting_k40_seed0_spatial", "stage2_k40_seed0"),
]


def rate(run: str) -> tuple[float, float, float, int] | None:
    f = REPO / "results" / run / "results.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())["summary"]
    lo, hi = wilson(s["n_success"], s["n_episodes"])
    return s["success_rate"], lo, hi, s["n_episodes"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="assets/forgetting.png")
    ap.add_argument("--json-out", default="results/forgetting.json")
    a = ap.parse_args()

    rows = []
    for label, old_run, new_run in BARS:
        old, new = rate(old_run), rate(new_run)
        if old is None or new is None:
            raise SystemExit(f"missing results for {label!r}")
        rows.append({"label": label, "old_line": old, "new_part": new,
                     "old_run": old_run, "new_run": new_run})

    x = range(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    old_y = [r["old_line"][0] * 100 for r in rows]
    new_y = [r["new_part"][0] * 100 for r in rows]
    old_err = [[(r["old_line"][0] - r["old_line"][1]) * 100 for r in rows],
               [(r["old_line"][2] - r["old_line"][0]) * 100 for r in rows]]
    new_err = [[(r["new_part"][0] - r["new_part"][1]) * 100 for r in rows],
               [(r["new_part"][2] - r["new_part"][0]) * 100 for r in rows]]

    ax.bar([i - w / 2 for i in x], old_y, w, yerr=old_err, capsize=4,
           color="#c53030", alpha=0.85, label="the old line (libero_spatial)")
    ax.bar([i + w / 2 for i in x], new_y, w, yerr=new_err, capsize=4,
           color="#2b6cb0", alpha=0.85, label="the new part (libero_object)")

    # Put the labels above the error bar, not on top of it.
    for i, r in enumerate(rows):
        for offset, key, err_hi in ((-w / 2, "old_line", old_err[1][i]),
                                    (w / 2, "new_part", new_err[1][i])):
            v = r[key][0] * 100
            ax.text(i + offset, v + err_hi + 2.5, f"{v:.0f}%",
                    ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_ylabel("task success rate  (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=9, loc="upper center")
    ax.set_title("Teaching a new part erases the old line", pad=12)
    fig.tight_layout()

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    (REPO / a.json_out).write_text(json.dumps(
        {"episodes_per_run": rows[0]["old_line"][3], "bars": rows}, indent=2) + "\n")

    for r in rows:
        print(f"{r['label'].replace(chr(10), ' '):<30} "
              f"old line {r['old_line'][0]*100:>5.1f}%   new part {r['new_part'][0]*100:>5.1f}%")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
