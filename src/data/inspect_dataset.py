#!/usr/bin/env python3
"""Phase 2: report what is actually inside `lerobot/libero`.

Numbers here drive the experiment design (how large K can be) and the operator
time model (how many seconds of human effort one demonstration represents), so
nothing is taken on trust: the suite membership of each task comes from LIBERO's
own benchmark registry rather than from string guessing, and episode lengths
come from the dataset's own metadata.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

# `lerobot/libero` covers the four standard benchmark suites, 10 tasks each.
# libero_90 is deliberately excluded: two of its 90 tasks share a language
# instruction with a task in libero_10 and libero_goal respectively
# ("pick up the book and place it in the back compartment of the caddy",
# "turn on the stove"). Including it makes the instruction ambiguous and,
# with a naive dict, silently relabels those two tasks.
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def suite_task_names() -> dict[str, str]:
    """language instruction -> suite, straight from LIBERO's benchmark registry.

    Refuses to guess: an instruction claimed by more than one of the four
    suites is an error rather than something to resolve by iteration order.
    """
    from libero.libero import benchmark

    claims: dict[str, set[str]] = {}
    for suite in SUITES:
        bench = benchmark.get_benchmark_dict()[suite]()
        for task in bench.tasks:
            claims.setdefault(task.language.strip().lower(), set()).add(suite)

    ambiguous = {k: sorted(v) for k, v in claims.items() if len(v) > 1}
    if ambiguous:
        raise RuntimeError(f"instruction claimed by multiple suites: {ambiguous}")
    return {k: next(iter(v)) for k, v in claims.items()}


def episode_task_index(data_dir: Path) -> pd.DataFrame:
    """episode_index -> task_index, read with column projection so it stays cheap."""
    frames = []
    for f in sorted(data_dir.rglob("*.parquet")):
        frames.append(pq.read_table(f, columns=["episode_index", "task_index"]).to_pandas())
    return pd.concat(frames).drop_duplicates("episode_index").set_index("episode_index")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--json-out", default="results/dataset_inventory.json")
    a = ap.parse_args()
    root = Path(a.dataset_root)

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    episodes = pd.read_parquet(root / "meta" / "episodes")[["episode_index", "length"]]
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet").reset_index()
    tasks.columns = ["task", "task_index"]
    ep_task = episode_task_index(root / "data")

    df = episodes.set_index("episode_index").join(ep_task).reset_index()
    df = df.merge(tasks, on="task_index", how="left")
    name2suite = suite_task_names()
    df["suite"] = df["task"].str.strip().str.lower().map(name2suite)

    print(f"dataset          : {root.name}")
    print(f"codebase_version : {info['codebase_version']}")
    print(f"fps (metadata)   : {fps}")
    print(f"episodes         : {info['total_episodes']}  (metadata) / {len(df)} (counted)")
    print(f"frames           : {info['total_frames']}  (metadata) / {int(df.length.sum())} (counted)")
    print(f"tasks            : {info['total_tasks']}  (metadata) / {df.task_index.nunique()} (counted)")

    unmapped = df[df.suite.isna()]["task"].unique()
    print(f"\ntasks not matched to a suite: {len(unmapped)}")
    for t in unmapped:
        print(f"  ! {t}")

    print("\n=== per suite ===")
    print(f"{'suite':<16} {'tasks':>6} {'episodes':>9} {'ep/task min':>12} {'max':>5} "
          f"{'frames/ep mean':>15} {'sec/ep mean':>12}")
    for suite, g in df.groupby("suite"):
        per_task = g.groupby("task_index").size()
        print(f"{suite:<16} {g.task_index.nunique():>6} {len(g):>9} {per_task.min():>12} "
              f"{per_task.max():>5} {g.length.mean():>15.1f} {g.length.mean() / fps:>12.2f}")

    print("\n=== libero_object, per task (this is the held-out suite) ===")
    obj = df[df.suite == "libero_object"].copy()
    print(f"{'task_index':>10} {'episodes':>9} {'len min':>8} {'mean':>7} {'max':>6} "
          f"{'sec mean':>9}  task")
    for ti, g in obj.groupby("task_index"):
        print(f"{ti:>10} {len(g):>9} {g.length.min():>8} {g.length.mean():>7.1f} "
              f"{g.length.max():>6} {g.length.mean()/fps:>9.2f}  {g.task.iloc[0][:52]}")

    per_task_counts = obj.groupby("task_index").size()
    payload = {
        "dataset_root": str(root),
        "revision": root.name,
        "fps_metadata": fps,
        "total_episodes": int(len(df)),
        "total_frames": int(df.length.sum()),
        "total_tasks": int(df.task_index.nunique()),
        "unmapped_tasks": list(unmapped),
        "per_suite": {
            str(s): {
                "n_tasks": int(g.task_index.nunique()),
                "n_episodes": int(len(g)),
                "episodes_per_task_min": int(g.groupby("task_index").size().min()),
                "episodes_per_task_max": int(g.groupby("task_index").size().max()),
                "frames_per_episode_mean": float(g.length.mean()),
                "frames_per_episode_min": int(g.length.min()),
                "frames_per_episode_max": int(g.length.max()),
                # Needed by Phase 3 to build the "factory's existing line" run.
                "episode_indices": sorted(int(x) for x in g.episode_index),
            }
            for s, g in df.groupby("suite")
        },
        "libero_object_per_task": {
            str(ti): {
                "task": g.task.iloc[0],
                "n_episodes": int(len(g)),
                "frames_min": int(g.length.min()),
                "frames_mean": float(g.length.mean()),
                "frames_max": int(g.length.max()),
                "episode_indices": sorted(int(x) for x in g.episode_index),
            }
            for ti, g in obj.groupby("task_index")
        },
        "max_feasible_K": int(per_task_counts.min()),
    }
    out = Path(a.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nmax feasible K (limited by the smallest task) = {payload['max_feasible_K']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
