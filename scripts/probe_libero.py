"""Tiny probe: bring up one LIBERO env and print what it actually returns.

Written before the real episode script so the episode script does not have to
guess observation keys or shapes.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  # noqa: E402

from lerobot.envs.libero import create_libero_envs  # noqa: E402

SUITE = "libero_object"

envs = create_libero_envs(
    task=SUITE,
    n_envs=1,
    gym_kwargs={"task_ids": [0]},
    env_cls=gym.vector.SyncVectorEnv,
    init_states=True,
    control_mode="relative",
)

print(f"\nreturned suites: {list(envs.keys())}")
task_ids = list(envs[SUITE].keys())
print(f"task_ids for {SUITE}: {task_ids}")

env = envs[SUITE][task_ids[0]]
print(f"\naction_space      = {env.action_space}")
print(f"observation_space = {env.observation_space}")

obs, info = env.reset(seed=42)
print(f"\nreset() obs type = {type(obs)}")
if isinstance(obs, dict):
    for k, v in obs.items():
        print(f"  {k:<40} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
print(f"reset() info keys = {list(info.keys()) if isinstance(info, dict) else type(info)}")

action = env.action_space.sample()
obs, rew, term, trunc, info = env.step(action)
print(f"\nstep() reward={rew} term={term} trunc={trunc}")
print(f"step() info keys = {list(info.keys()) if isinstance(info, dict) else type(info)}")

env.close()
print("\nprobe OK")
