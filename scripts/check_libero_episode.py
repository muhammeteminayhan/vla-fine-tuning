"""Phase 0 check 6: run one LIBERO episode end to end and record a video.

Random actions - this is not a policy test. The point is that the whole chain
(bddl task files -> robosuite -> MuJoCo -> EGL -> pixels -> mp4) runs start to
finish without error, and that the rollout rate is high enough to make Phase 4
evaluation affordable.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402

from lerobot.envs.libero import create_libero_envs  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_object")
p.add_argument("--task-id", type=int, default=0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--max-steps", type=int, default=300)
p.add_argument("--fps", type=int, default=20)  # robosuite control_freq
p.add_argument("--out", default="assets/phase0_libero_episode.mp4")
args = p.parse_args()

envs = create_libero_envs(
    task=args.suite,
    n_envs=1,
    gym_kwargs={"task_ids": [args.task_id]},
    env_cls=gym.vector.SyncVectorEnv,
    init_states=True,
    control_mode="relative",
)
env = envs[args.suite][args.task_id]


def agentview(obs) -> np.ndarray:
    """Pull the main camera frame out of the (batched, nested) observation."""
    img = obs["pixels"]["image"]
    return np.asarray(img)[0]  # drop the vector-env batch dim


rng = np.random.default_rng(args.seed)
obs, info = env.reset(seed=args.seed)

frames = [agentview(obs)]
steps = 0
success = False
terminated = truncated = False

t0 = time.perf_counter()
while steps < args.max_steps and not (terminated or truncated):
    action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
    obs, reward, term, trunc, info = env.step(action)
    terminated, truncated = bool(term[0]), bool(trunc[0])
    frames.append(agentview(obs))
    steps += 1
    if "is_success" in info:
        success = success or bool(np.asarray(info["is_success"]).ravel()[0])
elapsed = time.perf_counter() - t0

task_name = None
if "task" in info:
    t = np.asarray(info["task"]).ravel()
    task_name = str(t[0]) if t.size else None

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(out, frames, fps=args.fps, macro_block_size=1)

env.close()

result = {
    "suite": args.suite,
    "task_id": args.task_id,
    "task": task_name,
    "seed": args.seed,
    "steps": steps,
    "terminated": terminated,
    "truncated": truncated,
    "success": success,
    "frames_recorded": len(frames),
    "frame_shape": list(frames[0].shape),
    "wall_clock_s": round(elapsed, 3),
    "env_steps_per_s": round(steps / elapsed, 1),
    "sim_speed_vs_realtime": round((steps / args.fps) / elapsed, 2),
    "video": str(out),
    "video_bytes": out.stat().st_size,
}
print("\n=== EPISODE RESULT ===")
print(json.dumps(result, indent=2))
