#!/usr/bin/env bash
# Phase 3 overnight driver: stage-1 pre-training, then the K-shot runs.
#
# Ordering is seed-major on purpose. If this is interrupted, a complete K sweep
# at one seed is a usable curve; four partial seeds are not.
#
# Usage: scripts/run_phase3.sh <steps> <eval_episodes_per_task> [stage1_steps]
set -u
export MUJOCO_GL=egl

STEPS="${1:?usage: run_phase3.sh <steps> <eval_eps_per_task> [stage1_steps]}"
EVAL_EPS="${2:?}"
STAGE1_STEPS="${3:-8000}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAGE1_CKPT="checkpoints/stage1/checkpoints/last/pretrained_model"
PROGRESS="results/phase3_progress.log"

say() { echo "[$(date -Is)] $*" | tee -a "$PROGRESS"; }

# ---- stage 1: the factory's existing line -------------------------------
if [ ! -d "$STAGE1_CKPT" ]; then
  say "stage1 START steps=$STAGE1_STEPS"
  python src/train/run_train.py --split stage1 --steps "$STAGE1_STEPS" \
      --batch-size 16 --save-freq 0 --out-root checkpoints \
      >> "$PROGRESS" 2>&1 \
    && say "stage1 DONE" || { say "stage1 FAILED"; exit 1; }
else
  say "stage1 SKIP (already present)"
fi

[ -d "$STAGE1_CKPT" ] || { say "stage1 checkpoint missing at $STAGE1_CKPT"; exit 1; }

# ---- stage 2: K-shot runs, seed-major -----------------------------------
for SEED in 0 1 2; do
  for K in 5 10 20 40; do
    SPLIT="k${K}_seed${SEED}"
    CKPT="checkpoints/stage2/${SPLIT}/checkpoints/last/pretrained_model"

    if [ ! -d "$CKPT" ]; then
      say "train $SPLIT START steps=$STEPS"
      python src/train/run_train.py --split "$SPLIT" --steps "$STEPS" \
          --batch-size 16 --save-freq 0 \
          --init-from "$STAGE1_CKPT" --out-root checkpoints/stage2 \
          >> "$PROGRESS" 2>&1 \
        && say "train $SPLIT DONE" || { say "train $SPLIT FAILED"; continue; }
    else
      say "train $SPLIT SKIP (checkpoint present)"
    fi

    RUN="stage2_${SPLIT}"
    if [ ! -f "results/${RUN}/results.json" ]; then
      say "eval $SPLIT START"
      python src/eval/run_eval.py --policy "$CKPT" \
          --suite libero_object --n-episodes "$EVAL_EPS" --seed 1000 \
          --batch-size 1 --run-id "$RUN" >> "$PROGRESS" 2>&1 \
        && say "eval $SPLIT DONE" || say "eval $SPLIT FAILED"
    else
      say "eval $SPLIT SKIP (results present)"
    fi
  done
  say "=== seed $SEED complete ==="
done
say "PHASE 3 DRIVER FINISHED"
