#!/usr/bin/env python3
"""Phase 3 step 1: find what actually fits in 7.36 GiB, by measuring.

CLAUDE.md forbids guessing a batch size, and forbids silently lowering one after
an OOM. Here an OOM is the deliberate end of the search, so it is recorded as a
data point and reported - never worked around.

Two memory numbers are collected because they answer different questions:
  * `mem_gb` from lerobot's own log - what the training step allocates.
  * peak from nvidia-smi - what the process actually holds, including the CUDA
    context and allocator slack. This is the one that has to fit in the card.

Throughput comes from lerobot's logged `step_s`, taking the median over log
lines after the first, since the first includes dataloader warm-up.
"""

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STEP_RE = re.compile(r"step:(\d+).*?step_s:([\d.]+).*?mem_gb:([\d.]+)")


def run_one(bs: int, episodes: str, steps: int, out_dir: Path, log: Path,
            mixed_precision: str, grad_accum: int) -> dict:
    cmd = [
        "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        "--policy.input_features=null", "--policy.output_features=null",
        "--policy.optimizer_lr=1e-3", "--policy.scheduler_decay_lr=1e-4",
        "--policy.push_to_hub=false", "--policy.n_action_steps=10",
        "--dataset.repo_id=lerobot/libero",
        "--dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
        f"--dataset.episodes={episodes}",
        "--peft.method_type=LORA", "--peft.r=64", "--peft.lora_alpha=64",
        f"--accelerator.mixed_precision={mixed_precision}",
        f"--accelerator.gradient_accumulation.steps={grad_accum}",
        f"--batch_size={bs}", f"--steps={steps}", "--log_freq=10", "--num_workers=2",
        "--env_eval_freq=0", "--save_checkpoint=false", "--wandb.enable=false",
        "--seed=0", f"--output_dir={out_dir}",
    ]
    samples: list[int] = []
    t0 = time.perf_counter()
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
        while proc.poll() is None:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True)
            if r.stdout.strip().isdigit():
                samples.append(int(r.stdout.strip()))
            time.sleep(0.25)
        rc = proc.returncode
    wall = time.perf_counter() - t0

    text = log.read_text(errors="replace")
    oom = "CUDA out of memory" in text or "OutOfMemoryError" in text
    rows = STEP_RE.findall(text)
    step_times = [float(s) for _, s, _ in rows][1:]      # drop warm-up line
    mem_gbs = [float(m) for _, _, m in rows]

    return {
        "batch_size": bs,
        "mixed_precision": mixed_precision,
        "grad_accum": grad_accum,
        "exit_code": rc,
        "oom": oom,
        "ok": rc == 0 and not oom,
        "wall_clock_s": round(wall, 1),
        "gpu_peak_mib": max(samples) if samples else None,
        "lerobot_mem_gb": max(mem_gbs) if mem_gbs else None,
        "median_step_s": round(statistics.median(step_times), 4) if step_times else None,
        "steps_per_s": round(1 / statistics.median(step_times), 3) if step_times else None,
        "samples_per_s": round(bs / statistics.median(step_times), 2) if step_times else None,
        "log": str(log),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--split", default="k40_seed0")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", default="results/batch_size_sweep/results.json")
    a = ap.parse_args()

    splits = json.loads((REPO / "configs" / "kshot_splits.json").read_text())
    episodes = json.dumps(splits["splits"][a.split]["episodes"])
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    results = []
    for bs in a.batch_sizes:
        print(f"--- batch_size={bs} (mp={a.mixed_precision}, accum={a.grad_accum}) ...", flush=True)
        r = run_one(bs, episodes, a.steps, scratch / f"bs{bs}", scratch / f"bs{bs}.log",
                    a.mixed_precision, a.grad_accum)
        results.append(r)
        status = "OOM" if r["oom"] else ("OK" if r["ok"] else f"FAIL rc={r['exit_code']}")
        print(f"    {status}  peak={r['gpu_peak_mib']} MiB  step_s={r['median_step_s']}  "
              f"samples/s={r['samples_per_s']}", flush=True)
        if r["oom"] or not r["ok"]:
            print("    stopping sweep here; an OOM is a result, not something to work around.")
            break

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": a.split, "steps_per_run": a.steps,
        "mixed_precision": a.mixed_precision, "grad_accum": a.grad_accum,
        "vram_total_mib": 7527, "vram_free_at_idle_mib": 7355,
        "runs": results,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
