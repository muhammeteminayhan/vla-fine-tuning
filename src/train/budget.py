#!/usr/bin/env python3
"""Phase 3: turn the measured throughput into a wall-clock budget for the study.

Phase 2 flagged the compute cost as an open risk with no measurement behind it.
The batch-size sweep supplies the missing number, so the budget below is derived
from measurements in results/batch_size_sweep/results.json rather than from the
LeRobot documentation's A100 figures.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sweep = json.loads((REPO / "results/batch_size_sweep/results.json").read_text())
splits = json.loads((REPO / "configs/kshot_splits.json").read_text())
inv = json.loads((REPO / "results/dataset_inventory.json").read_text())

ok = [r for r in sweep["runs"] if r["ok"]]
print("=== measured throughput (bf16, LoRA r=64) ===")
for r in ok:
    print(f"  bs={r['batch_size']:>3}  peak={r['gpu_peak_mib']:>5} MiB  "
          f"step_s={r['median_step_s']:<7} samples/s={r['samples_per_s']}")
oom = [r for r in sweep["runs"] if r["oom"]]
for r in oom:
    print(f"  bs={r['batch_size']:>3}  OOM (reached {r['gpu_peak_mib']} MiB of "
          f"{sweep['vram_total_mib']} MiB)")

# Throughput is flat from bs=4 upward, so use the plateau value.
plateau = [r for r in ok if r["batch_size"] >= 4]
sps = sum(r["samples_per_s"] for r in plateau) / len(plateau)
print(f"\nplateau samples/s (bs>=4): {sps:.2f}")

# Dataset sizes actually seen by each run.
obj = inv["per_suite"]["libero_object"]["frames_per_episode_mean"]
k_frames = {k: round(10 * k * obj) for k in splits["k_values"]}
stage1_frames = sum(inv["per_suite"][s]["n_episodes"] * inv["per_suite"][s]["frames_per_episode_mean"]
                    for s in ["libero_spatial", "libero_goal", "libero_10"])
print("\n=== dataset size per run (frames) ===")
for k, f in k_frames.items():
    print(f"  K={k:<3} {f:>8,}")
print(f"  stage1 {stage1_frames:>8,.0f}")


def hours(samples: float) -> float:
    return samples / sps / 3600


print(f"\n=== scenarios (3 seeds x 4 K values = 12 stage-2 runs) ===")
BS = 16
scenarios = {
    "docs default (100k steps)": 100_000,
    "20k steps": 20_000,
    "10k steps": 10_000,
    "5k steps": 5_000,
}
print(f"assuming batch_size={BS}\n")
print(f"| {'steps per run':<26} | {'h / run':>8} | {'12 runs':>9} | {'+ stage1':>9} | {'total':>8} |")
print(f"|{'-'*28}|{'-'*10}|{'-'*11}|{'-'*11}|{'-'*10}|")
rows = {}
for name, steps in scenarios.items():
    h = hours(steps * BS)
    stage1_h = hours(steps * 2 * BS)   # stage 1 sees ~3.5x the data; give it 2x the steps
    total = 12 * h + stage1_h
    rows[name] = {"steps": steps, "h_per_run": round(h, 2),
                  "h_12_runs": round(12 * h, 1), "h_stage1": round(stage1_h, 1),
                  "h_total": round(total, 1)}
    print(f"| {name:<26} | {h:>8.2f} | {12*h:>9.1f} | {stage1_h:>9.1f} | {total:>8.1f} |")

print("\n=== epochs each scenario implies (how many times each demo is seen) ===")
print(f"| {'steps':>7} | " + " | ".join(f"K={k:<3}" for k in k_frames) + " |")
for name, steps in scenarios.items():
    eps = [f"{steps*BS/f:>5.0f}" for f in k_frames.values()]
    print(f"| {steps:>7} | " + " | ".join(eps) + " |")

out = REPO / "results/compute_budget.json"
out.write_text(json.dumps({
    "source": "results/batch_size_sweep/results.json",
    "plateau_samples_per_s": round(sps, 2),
    "assumed_batch_size": BS,
    "frames_per_run": k_frames,
    "stage1_frames": round(stage1_frames),
    "scenarios": rows,
}, indent=2) + "\n")
print(f"\nwrote {out}")
