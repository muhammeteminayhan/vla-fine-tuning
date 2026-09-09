#!/usr/bin/env python3
"""Phase 4: turn failed rollouts into something a human can actually review.

CLAUDE.md asks for failures to be watched and classified (wrong object, missed
grasp, early release, missed target). Opening dozens of short videos one at a
time is the slow way to do that. This lays each failure out as a single strip of
evenly spaced frames, so a whole task's failure modes can be compared at a
glance and the classification can be done from one image per episode.
"""

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def strip(video: Path, out: Path, n: int, height: int) -> bool:
    """Tile n evenly spaced frames of `video` into one horizontal image."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True).stdout.strip()
    if not probe.isdigit():
        return False
    total = int(probe)
    step = max(1, total // n)
    # select every `step`-th frame, cap at n, scale, then tile in one row
    vf = (f"select='not(mod(n\\,{step}))',scale=-1:{height},tile={n}x1")
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", vf,
         "-frames:v", "1", str(out)],
        capture_output=True, text=True)
    return r.returncode == 0 and out.exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="results/<run_id> directory")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--max-per-task", type=int, default=2,
                    help="strips per task, to keep the review set small")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    run = Path(a.run)
    results = json.loads((run / "results.json").read_text())
    out_dir = Path(a.out_dir) if a.out_dir else run / "failure_strips"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task: dict[int, int] = {}
    made = []
    for ep in results["episodes"]:
        if ep["success"] or not ep.get("video"):
            continue
        t = ep["task_id"]
        if per_task.get(t, 0) >= a.max_per_task:
            continue
        video = run / "lerobot_raw" / ep["video"]
        if not video.exists():
            continue
        out = out_dir / f"task{t:02d}_ep{ep['episode_ix']}_seed{ep['seed']}.jpg"
        if strip(video, out, a.frames, a.height):
            per_task[t] = per_task.get(t, 0) + 1
            made.append({"task_id": t, "episode_ix": ep["episode_ix"],
                         "seed": ep["seed"], "strip": str(out)})

    index = out_dir / "index.json"
    index.write_text(json.dumps(
        {"run": str(run), "n_strips": len(made), "frames_per_strip": a.frames,
         "strips": made}, indent=2) + "\n")
    print(f"{len(made)} strips over {len(per_task)} tasks -> {out_dir}")
    for m in made:
        print(f"  task {m['task_id']:>2}  ep {m['episode_ix']}  {Path(m['strip']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
