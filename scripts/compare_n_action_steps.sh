#!/usr/bin/env bash
# Phase 1 decision experiment: n_action_steps 50 (SmolVLA default) vs 10
# (what the LIBERO docs use for Pi0.5). Everything else is held fixed.
set -u
export MUJOCO_GL=egl

OUT="${1:?usage: compare_n_action_steps.sh <output_root>}"
mkdir -p "$OUT"

RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

for N in 50 10; do
  echo "### n_action_steps=$N ###"
  rm -rf "$OUT/nas${N}"
  scripts/run_with_gpu_peak.sh "n_action_steps=$N" "$OUT/nas${N}.log" \
    lerobot-eval \
      --policy.path=lerobot/smolvla_base \
      --policy.n_action_steps="$N" \
      --env.type=libero \
      --env.task=libero_object \
      --env.task_ids='[0,1]' \
      --env.control_mode=relative \
      --env.init_states=true \
      --env.max_parallel_tasks=1 \
      --eval.n_episodes=5 \
      --eval.batch_size=1 \
      --seed=1000 \
      --rename_map="$RENAME" \
      --output_dir="$OUT/nas${N}" \
    | tee "$OUT/nas${N}.metrics.json"
done
