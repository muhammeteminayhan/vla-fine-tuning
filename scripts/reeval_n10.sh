#!/usr/bin/env bash
# Phase 4: re-evaluate every checkpoint at the published LIBERO resolution,
# 10 episodes per task instead of 5.
#
# Seeds are start_seed + episode_ix, so episodes 0-4 repeat the runs already on
# disk bit for bit (Phase 1 proved the determinism) and 5-9 are new. Results go
# to separate *_n10 run ids so both resolutions stay traceable.
set -u
export MUJOCO_GL=egl
cd "$(dirname "$0")/.."
LOG=results/reeval_n10_progress.log
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for any running eval to finish"
while pgrep -f 'src/eval/run_eval.py' > /dev/null; do sleep 30; done
say "starting"

run() {  # run <run_id> <checkpoint> [extra args]
  local id="$1"; shift
  local ckpt="$1"; shift
  if [ -f "results/${id}/results.json" ]; then say "$id SKIP"; return; fi
  say "$id START"
  python src/eval/run_eval.py --policy "$ckpt" --suite libero_object \
      --n-episodes 10 --seed 1000 --batch-size 1 --run-id "$id" "$@" >> "$LOG" 2>&1 \
    && say "$id DONE" || say "$id FAILED"
}

run stage1_baseline_n10 checkpoints/stage1/checkpoints/last/pretrained_model
for SEED in 0 1 2; do
  for K in 5 10 20 40; do
    run "stage2_k${K}_seed${SEED}_n10" \
        "checkpoints/stage2/k${K}_seed${SEED}/checkpoints/last/pretrained_model"
  done
done
say "RE-EVAL FINISHED"
