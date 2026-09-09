#!/usr/bin/env python3
"""Phase 4: the teaching cost curve.

x is K, the number of demonstrations per new task. Operator minutes appear as a
derived top axis rather than the primary one, because ~73% of that figure comes
from an assumed reset time (docs/02-experiment-design.md §4). Plotting minutes
as the primary axis would present an assumption with the authority of a
measurement. The curve's shape is identical either way - only the labelling
changes.

Two uncertainty representations, answering different questions:
  * shaded band  - Wilson interval over the pooled episodes at each K
  * error bars   - spread across training seeds, when there are at least two
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - margin), min(1.0, centre + margin)


def collect(pattern: str) -> dict[int, list[dict]]:
    """K -> list of per-seed run records."""
    out = defaultdict(list)
    for p in sorted(REPO.glob(pattern)):
        f = p / "results.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        k = d.get("K")
        if k is None:                      # infer from the run id, e.g. stage2_k20_seed1
            for part in p.name.split("_"):
                if part.startswith("k") and part[1:].isdigit():
                    k = int(part[1:])
        if k is None:
            continue
        out[int(k)].append(d)
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="results/stage2_k*_seed*")
    ap.add_argument("--baseline", default=None,
                    help="run dir for the K=0 reference line")
    ap.add_argument("--out", default="assets/teaching_cost_curve.png")
    ap.add_argument("--json-out", default="results/teaching_cost_curve.json")
    a = ap.parse_args()

    splits = json.loads((REPO / "configs" / "kshot_splits.json").read_text())
    demo_s = splits["suite_mean_demo_seconds"]
    reset_s = splits["reset_seconds_assumed"]
    per_demo_s = demo_s + reset_s

    runs = collect(a.pattern)
    if not runs:
        raise SystemExit(f"no runs matched {a.pattern!r}")

    points = []
    for k in sorted(runs):
        rs = runs[k]
        n = sum(r["summary"]["n_episodes"] for r in rs)
        s = sum(r["summary"]["n_success"] for r in rs)
        lo, hi = wilson(s, n)
        seed_rates = [r["summary"]["success_rate"] for r in rs]
        points.append({
            "K": k, "n_seeds": len(rs), "n_episodes": n, "n_success": s,
            "success_rate": s / n, "wilson_lo": lo, "wilson_hi": hi,
            "seed_rates": seed_rates,
            "seed_min": min(seed_rates), "seed_max": max(seed_rates),
            "operator_minutes_per_task": round(k * per_demo_s / 60, 2),
        })

    base = None
    if a.baseline and (REPO / a.baseline / "results.json").exists():
        d = json.loads((REPO / a.baseline / "results.json").read_text())["summary"]
        blo, bhi = wilson(d["n_success"], d["n_episodes"])
        base = {"success_rate": d["success_rate"], "n_episodes": d["n_episodes"],
                "wilson_lo": blo, "wilson_hi": bhi}

    ks = [p["K"] for p in points]
    ys = [p["success_rate"] * 100 for p in points]
    lo = [p["wilson_lo"] * 100 for p in points]
    hi = [p["wilson_hi"] * 100 for p in points]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.fill_between(ks, lo, hi, alpha=0.18, color="#2b6cb0",
                    label="Wilson 95% CI (pooled episodes)")
    ax.plot(ks, ys, "o-", color="#2b6cb0", lw=2, ms=7, label="SmolVLA + LoRA")

    if any(p["n_seeds"] >= 2 for p in points):
        err_lo = [max(0, p["success_rate"] - p["seed_min"]) * 100 for p in points]
        err_hi = [max(0, p["seed_max"] - p["success_rate"]) * 100 for p in points]
        ax.errorbar(ks, ys, yerr=[err_lo, err_hi], fmt="none",
                    ecolor="#2b6cb0", capsize=4, alpha=0.8,
                    label="spread across seeds")

    if base:
        ax.axhline(base["success_rate"] * 100, ls="--", color="#c53030", lw=1.5,
                   label=f"K=0 baseline ({base['success_rate']*100:.0f}%)")
        ax.axhspan(base["wilson_lo"] * 100, base["wilson_hi"] * 100,
                   color="#c53030", alpha=0.08)

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    # A log axis decorates itself with minor ticks like "6 x 10^0"; the only
    # meaningful x values here are the K we actually ran.
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("demonstrations per new task  (K)")
    ax.set_ylabel("task success rate  (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(loc="lower right", fontsize=9)

    top = ax.secondary_xaxis("top")
    top.set_xticks(ks)
    top.set_xticklabels([f"{k * per_demo_s / 60:.1f}" for k in ks])
    top.xaxis.set_minor_locator(NullLocator())
    top.xaxis.set_minor_formatter(NullFormatter())
    top.set_xlabel(f"operator minutes per new task  "
                   f"(demo {demo_s:.1f}s measured + reset {reset_s:.0f}s ASSUMED)",
                   fontsize=9)

    ax.set_title("Teaching cost curve: SmolVLA on unseen LIBERO-Object tasks", pad=28)
    fig.tight_layout()
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)

    payload = {"points": points, "baseline": base,
               "operator_time_model": {"demo_seconds_measured": demo_s,
                                       "reset_seconds_assumed": reset_s,
                                       "seconds_per_demo": per_demo_s},
               "figure": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out)}
    (REPO / a.json_out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'K':>4} {'seeds':>6} {'episodes':>9} {'success':>8} {'Wilson 95%':>20} {'op.min':>8}")
    for p in points:
        print(f"{p['K']:>4} {p['n_seeds']:>6} {p['n_episodes']:>9} "
              f"{p['success_rate']*100:>7.1f}% "
              f"[{p['wilson_lo']*100:>5.1f}%,{p['wilson_hi']*100:>6.1f}%] "
              f"{p['operator_minutes_per_task']:>8.1f}")
    if base:
        print(f"baseline K=0: {base['success_rate']*100:.1f}% over {base['n_episodes']} episodes")
    print(f"\nwrote {out}\nwrote {REPO / a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
