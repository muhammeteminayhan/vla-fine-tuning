#!/usr/bin/env python3
"""Phase 1 evaluation harness: a deterministic wrapper around `lerobot-eval`.

Why this exists rather than calling `lerobot-eval` directly:

  * `eval_info.json` does not record per-episode seeds - `eval_one` drops the
    field. We reconstruct them as `seed = start_seed + episode_ix`, a formula
    verified empirically by scripts/verify_seed_mapping.py.
  * `EvalConfig.batch_size` defaults to 0, which means "auto-tune from CPU core
    count". That makes results machine-dependent. Here it is always explicit.
  * Nothing upstream records the git commit, the policy path, or the machine the
    numbers came from, and Phase 4 requires every reported number to be
    traceable.
  * Episode length is absent from `eval_info.json` (`return_episode_data` is
    hardcoded False), so we recover it from the rendered video frame count.
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# smolvla_base declares camera1/2/3; LIBERO provides image/image2. Needed only
# for the un-finetuned base model (the K=0 reference line).
BASE_MODEL_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def git_info() -> dict:
    return {
        "commit": sh(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]) or None,
        "dirty": bool(sh(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])),
        "lerobot_commit": sh(["git", "-C", str(Path.home() / "lerobot"), "rev-parse", "HEAD"]) or None,
    }


def system_info() -> dict:
    import torch

    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
    }


def video_frames(path: Path) -> int | None:
    """Episode length in env steps, recovered from the rendered video."""
    if not path.exists():
        return None
    out = sh([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
    ])
    return int(out) if out.isdigit() else None


def build_command(a, out_dir: Path) -> list[str]:
    cmd = [
        "lerobot-eval",
        f"--policy.path={a.policy}",
        f"--policy.n_action_steps={a.n_action_steps}",
        "--env.type=libero",
        f"--env.task={a.suite}",
        f"--env.control_mode={a.control_mode}",
        "--env.init_states=true",
        "--env.hard_reset=true",          # bit-identical resets; required for the determinism gate
        "--env.max_parallel_tasks=1",
        f"--eval.n_episodes={a.n_episodes}",
        f"--eval.batch_size={a.batch_size}",   # never left at 0 (= CPU-count dependent)
        f"--seed={a.seed}",
        f"--output_dir={out_dir}",
    ]
    if a.task_ids:
        cmd.append(f"--env.task_ids=[{','.join(str(t) for t in a.task_ids)}]")
    if a.base_model:
        cmd.append(f"--rename_map={json.dumps(BASE_MODEL_RENAME_MAP)}")
    return cmd


def collect_episodes(eval_info: dict, out_dir: Path, start_seed: int) -> list[dict]:
    episodes = []
    for task in eval_info.get("per_task", []):
        suite, task_id, m = task["task_group"], task["task_id"], task["metrics"]
        videos = m.get("video_paths") or []
        for ix, success in enumerate(m["successes"]):
            video = Path(videos[ix]) if ix < len(videos) else None
            episodes.append({
                "suite": suite,
                "task_id": task_id,
                "episode_ix": ix,
                "seed": start_seed + ix,       # verified by scripts/verify_seed_mapping.py
                "success": bool(success),
                "sum_reward": m["sum_rewards"][ix],
                "max_reward": m["max_rewards"][ix],
                "episode_length": video_frames(video) if video else None,
                "video": str(video.relative_to(out_dir)) if video and video.exists() else None,
            })
    return episodes


def index_failures(episodes: list[dict], run_dir: Path, raw_dir: Path) -> int:
    """Symlink failed rollouts under failures/ with self-describing names.

    Phase 4 has to watch these and classify what went wrong (wrong object,
    missed grasp, early release, missed target). Hunting for them inside
    lerobot's per-task directory tree by episode index is a waste of time, so
    name them by what they are.
    """
    fail_dir = run_dir / "failures"
    fail_dir.mkdir(exist_ok=True)
    n = 0
    for ep in episodes:
        if ep["success"] or not ep["video"]:
            continue
        src = (raw_dir / ep["video"]).resolve()
        if not src.exists():
            continue
        link = fail_dir / f"{ep['suite']}_task{ep['task_id']}_ep{ep['episode_ix']}_seed{ep['seed']}.mp4"
        link.unlink(missing_ok=True)
        link.symlink_to(src)
        n += 1
    return n


def summarize(episodes: list[dict]) -> dict:
    per_task = {}
    for ep in episodes:
        per_task.setdefault(f"{ep['suite']}/{ep['task_id']}", []).append(ep["success"])
    return {
        "n_episodes": len(episodes),
        "n_success": sum(e["success"] for e in episodes),
        "success_rate": (sum(e["success"] for e in episodes) / len(episodes)) if episodes else None,
        "per_task_success_rate": {k: sum(v) / len(v) for k, v in sorted(per_task.items())},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, help="checkpoint path or hub id")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-action-steps", type=int, default=10, help="fixed project-wide; see CLAUDE.md")
    p.add_argument("--control-mode", default="relative", choices=["relative", "absolute"])
    p.add_argument("--base-model", action="store_true", help="un-finetuned smolvla_base: adds rename_map")
    p.add_argument("--run-id", default=None)
    p.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    run_id = a.run_id or f"{a.suite}_seed{a.seed}_nas{a.n_action_steps}_{datetime.datetime.now():%Y%m%dT%H%M%S}"
    run_dir = Path(a.results_root) / run_id
    raw_dir = run_dir / "lerobot_raw"
    cmd = build_command(a, raw_dir)

    if a.dry_run:
        print(" \\\n  ".join(cmd))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now().astimezone()
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    ended = datetime.datetime.now().astimezone()

    eval_info_path = raw_dir / "eval_info.json"
    if rc != 0 or not eval_info_path.exists():
        print(f"FAILED: exit={rc}, eval_info.json present={eval_info_path.exists()}", file=sys.stderr)
        return rc or 1

    eval_info = json.loads(eval_info_path.read_text())
    episodes = collect_episodes(eval_info, raw_dir, a.seed)
    n_failures_indexed = index_failures(episodes, run_dir, raw_dir)

    result = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "wall_clock_s": round((ended - started).total_seconds(), 2),
        "git": git_info(),
        "system": system_info(),
        "policy": {"path": a.policy, "n_action_steps": a.n_action_steps, "is_base_model": a.base_model},
        "env": {"type": "libero", "suite": a.suite, "task_ids": a.task_ids,
                "control_mode": a.control_mode, "init_states": True, "hard_reset": True},
        "eval": {"n_episodes": a.n_episodes, "batch_size": a.batch_size, "start_seed": a.seed},
        "command": cmd,
        "episodes": episodes,
        "summary": summarize(episodes),
        "failures_indexed": n_failures_indexed,
        "notes": {
            "seed_derivation": "seed = start_seed + episode_ix; verified by scripts/verify_seed_mapping.py",
            "episode_length_source": "frame count of the rendered video (ffprobe -count_frames)",
            "failure_videos": "symlinked under failures/ for Phase 4 error analysis",
        },
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {run_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
