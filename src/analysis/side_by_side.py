#!/usr/bin/env python3
"""Phase 5: the before/after clip for the README.

Left is the factory's own model on a part it has never seen; right is the same
model after the operator demonstrated that part K times. Same task, same seed,
so the two rollouts start from an identical scene and the only difference is the
teaching.

The task is chosen to sit near the middle of the difficulty range rather than at
the easy end - picking the task the model is best at would make the clip a
better advertisement and a worse measurement.
"""

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def find(run: str, task_id: int, seed: int) -> tuple[Path, bool] | None:
    f = REPO / "results" / run / "results.json"
    if not f.exists():
        return None
    for e in json.loads(f.read_text())["episodes"]:
        if e["task_id"] == task_id and e["seed"] == seed and e.get("video"):
            return REPO / "results" / run / "lerobot_raw" / e["video"], e["success"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before-run", default="stage1_baseline")
    ap.add_argument("--after-run", default="stage2_k20_seed1")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="assets/before_after_k20.mp4")
    a = ap.parse_args()

    before = find(a.before_run, a.task_id, a.seed)
    after = find(a.after_run, a.task_id, a.seed)
    if not before or not after:
        raise SystemExit(f"missing episode for task {a.task_id} seed {a.seed}")
    (bv, bs), (av, asucc) = before, after
    if bs or not asucc:
        raise SystemExit(f"expected before=failure after=success, got {bs} / {asucc}")

    inv = json.loads((REPO / "results/dataset_inventory.json").read_text())
    per = inv["libero_object_per_task"]
    name = per[sorted(per, key=int)[a.task_id]]["task"]

    def esc(s: str) -> str:
        """drawtext parses ':' and ',' as option separators even inside quotes."""
        for ch in ("\\", "'", ":", ",", "%", "[", "]"):
            s = s.replace(ch, "\\" + ch)
        return s

    def label(text: str, y: str) -> str:
        return ("drawtext=fontcolor=white:fontsize=18:box=1:boxcolor=black@0.65:"
                f"boxborderw=8:x=(w-text_w)/2:y={y}:text='{esc(text)}'")

    def nframes(v: Path) -> int:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(v)],
            capture_output=True, text=True).stdout.strip()
        return int(out) if out.isdigit() else 0

    # The failure runs to the 280-step cap while the success finishes early.
    # Hold the shorter clip on its last frame rather than looping it, so the
    # difference in how long each rollout took stays visible.
    nb, na = nframes(bv), nframes(av)
    pad_l = max(0, na - nb) / a.fps
    pad_r = max(0, nb - na) / a.fps
    tpad_l = f",tpad=stop_mode=clone:stop_duration={pad_l}" if pad_l else ""
    tpad_r = f",tpad=stop_mode=clone:stop_duration={pad_r}" if pad_r else ""

    vf = (
        f"[0:v]scale=360:360{tpad_l},{label('BEFORE - factory model, part never seen', 'h-32')}[l];"
        f"[1:v]scale=360:360{tpad_r},"
        f"{label('AFTER - 20 demonstrations (9 operator min)', 'h-32')}[r];"
        f"[l][r]hstack=inputs=2:shortest=1,pad=iw:ih+36:0:36:black,{label(name, '9')}"
    )

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(bv), "-i", str(av),
           "-filter_complex", vf,
           "-r", str(a.fps), "-pix_fmt", "yuv420p", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        return 1
    print(f"task {a.task_id} seed {a.seed}: {name}")
    print(f"  before : {bv.relative_to(REPO)}  success={bs}")
    print(f"  after  : {av.relative_to(REPO)}  success={asucc}")
    print(f"  frames : before={nb} after={na}")
    print(f"  wrote  : {out.relative_to(REPO)} ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
