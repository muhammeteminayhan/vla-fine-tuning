"""Verify, do not assume: how does lerobot-eval map episodes to seeds?

`eval_info.json` does not contain per-episode seeds - `eval_one` drops the field
when it collapses `per_episode` into `TaskMetrics`. Our harness therefore has to
reconstruct them. Reading the source suggests episode i gets `start_seed + i`
(lerobot_eval.py:528-532), but a schema built on an unverified assumption is a
schema that lies.

So: spy on `eval_policy`, which still has the seeds, and check the formula.
"""

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import lerobot.scripts.lerobot_eval as LE  # noqa: E402

START_SEED = 1000
N_EPISODES = 4
BATCH_SIZE = 1

captured = []
_orig_eval_policy = LE.eval_policy


def spy(*args, **kwargs):
    result = _orig_eval_policy(*args, **kwargs)
    captured.append(
        {
            "start_seed": kwargs.get("start_seed"),
            "per_episode": [
                {"episode_ix": ep["episode_ix"], "seed": ep["seed"], "success": ep["success"]}
                for ep in result["per_episode"]
            ],
        }
    )
    return result


LE.eval_policy = spy

RENAME = json.dumps(
    {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
)

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/verify_seed_mapping"

sys.argv = [
    "lerobot-eval",
    "--policy.path=lerobot/smolvla_base",
    "--policy.n_action_steps=10",
    "--env.type=libero",
    "--env.task=libero_object",
    "--env.task_ids=[0,1]",
    "--env.control_mode=relative",
    "--env.init_states=true",
    "--env.max_parallel_tasks=1",
    f"--eval.n_episodes={N_EPISODES}",
    f"--eval.batch_size={BATCH_SIZE}",
    f"--seed={START_SEED}",
    f"--rename_map={RENAME}",
    f"--output_dir={OUT_DIR}",
]

LE.eval_main()

print("\n" + "=" * 70)
print("SEED MAPPING VERIFICATION")
print("=" * 70)
print(f"start_seed = {START_SEED}, n_episodes = {N_EPISODES}, batch_size = {BATCH_SIZE}")
print(f"tasks captured = {len(captured)}\n")

expected = [START_SEED + i for i in range(N_EPISODES)]
all_ok = True
for ti, cap in enumerate(captured):
    actual = [ep["seed"] for ep in cap["per_episode"]]
    ixs = [ep["episode_ix"] for ep in cap["per_episode"]]
    ok = actual == expected
    all_ok &= ok
    print(f"task #{ti}: start_seed passed in = {cap['start_seed']}")
    print(f"  episode_ix = {ixs}")
    print(f"  seeds      = {actual}")
    print(f"  expected   = {expected}   -> {'MATCH' if ok else 'MISMATCH'}")

print()
print(f"formula 'seed = start_seed + episode_ix' holds: {all_ok}")
seed_sets = [[ep["seed"] for ep in c["per_episode"]] for c in captured]
print(f"all tasks use the SAME seed sequence: {len(set(map(tuple, seed_sets))) == 1}")
