#!/usr/bin/env python3
"""Phase 3: run one K-shot fine-tune and record everything needed to trust it.

CLAUDE.md requires each run to log K, seed, step count, wall-clock, peak VRAM,
final loss and checkpoint path. It also forbids silently changing configuration
between runs, so the LoRA settings live here as module constants rather than as
command-line defaults that could drift from run to run: every K uses the same
values, and the values that were actually used are written into the run record.

Peak VRAM is sampled from nvidia-smi rather than read from the training log,
because what has to fit on the card includes the CUDA context and allocator
slack, not just what the step allocates.
"""

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"

# Fixed for every run in the study. Changing any of these invalidates comparisons.
LORA = {"method_type": "LORA", "r": 64, "lora_alpha": 64}
OPTIM = {"optimizer_lr": 1e-3, "scheduler_decay_lr": 1e-4}
N_ACTION_STEPS = 10          # decided in Phase 1, see CLAUDE.md
MIXED_PRECISION = "bf16"

STEP_RE = re.compile(r"step:(\d+).*?loss:([\d.]+).*?step_s:([\d.]+).*?mem_gb:([\d.]+)")


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="key in configs/kshot_splits.json, e.g. k40_seed0")
    ap.add_argument("--init-from", default="lerobot/smolvla_base",
                    help="stage-2 runs start from the stage-1 checkpoint")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-freq", type=int, default=0, help="0 = only the final checkpoint")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out-root", default="checkpoints")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    splits = json.loads((REPO / "configs" / "kshot_splits.json").read_text())
    if a.split not in splits["splits"]:
        raise SystemExit(f"unknown split {a.split!r}; have {sorted(splits['splits'])}")
    spec = splits["splits"][a.split]
    episodes = json.dumps(spec["episodes"])

    out_dir = REPO / a.out_root / a.split
    cmd = [
        "lerobot-train",
        f"--policy.path={a.init_from}",
        "--policy.input_features=null", "--policy.output_features=null",
        f"--policy.optimizer_lr={OPTIM['optimizer_lr']}",
        f"--policy.scheduler_decay_lr={OPTIM['scheduler_decay_lr']}",
        "--policy.push_to_hub=false",
        f"--policy.n_action_steps={N_ACTION_STEPS}",
        "--dataset.repo_id=lerobot/libero",
        f"--dataset.revision={DATASET_REVISION}",
        f"--dataset.episodes={episodes}",
        f"--peft.method_type={LORA['method_type']}",
        f"--peft.r={LORA['r']}", f"--peft.lora_alpha={LORA['lora_alpha']}",
        f"--accelerator.mixed_precision={MIXED_PRECISION}",
        f"--batch_size={a.batch_size}", f"--steps={a.steps}",
        f"--save_freq={a.save_freq}", "--save_checkpoint=true",
        f"--num_workers={a.num_workers}",
        "--env_eval_freq=0", "--wandb.enable=false", "--log_freq=50",
        f"--seed={spec['seed']}", f"--output_dir={out_dir}",
    ]
    if a.dry_run:
        print(" \\\n  ".join(cmd))
        return 0

    # lerobot-train refuses to start if output_dir already exists, so it must own
    # the directory. The log therefore lives beside it until the run finishes.
    if out_dir.exists():
        raise SystemExit(
            f"{out_dir} already exists. Refusing to overwrite a previous run - "
            f"delete it explicitly if that is what you want."
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"{out_dir.name}.train.log"
    samples: list[int] = []
    started = datetime.now().astimezone()
    t0 = time.perf_counter()
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
        while proc.poll() is None:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if r.stdout.strip().isdigit():
                samples.append(int(r.stdout.strip()))
            time.sleep(0.5)
        rc = proc.returncode
    wall = time.perf_counter() - t0

    text = log.read_text(errors="replace")
    rows = STEP_RE.findall(text)
    oom = "CUDA out of memory" in text or "OutOfMemoryError" in text
    ckpts = sorted(p for p in (out_dir / "checkpoints").glob("*") if p.is_dir()) \
        if (out_dir / "checkpoints").exists() else []

    record = {
        "split": a.split, "K": spec["K"], "seed": spec["seed"],
        "init_from": a.init_from,
        "steps": a.steps, "batch_size": a.batch_size,
        "effective_samples": a.steps * a.batch_size,
        "n_episodes": spec["n_episodes_total"],
        "lora": LORA, "optimizer": OPTIM,
        "n_action_steps": N_ACTION_STEPS, "mixed_precision": MIXED_PRECISION,
        "dataset_revision": DATASET_REVISION,
        "started_at": started.isoformat(),
        "wall_clock_s": round(wall, 1),
        "wall_clock_h": round(wall / 3600, 3),
        "exit_code": rc, "oom": oom, "ok": rc == 0 and not oom,
        "gpu_peak_mib": max(samples) if samples else None,
        "final_loss": float(rows[-1][1]) if rows else None,
        "loss_first_logged": float(rows[0][1]) if rows else None,
        "median_step_s": None if not rows else
            sorted(float(r[2]) for r in rows[1:])[max(0, (len(rows) - 1) // 2)],
        "checkpoints": [str(p.relative_to(REPO)) for p in ckpts],
        "git_commit": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]) or None,
        "command": cmd,
    }
    if out_dir.exists():                 # move the log in beside the checkpoints
        log = log.rename(out_dir / "train.log")
        record["log"] = str(log.relative_to(REPO))
        (out_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n")
    else:
        record["log"] = str(log.relative_to(REPO))
        log.with_suffix(".run_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "command"}, indent=2))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
