#!/usr/bin/env bash
# Phase 0 gate: run all six environment checks and dump raw output.
# Usage:  bash scripts/run_phase0_checks.sh 2>&1 | tee results/phase0_checks.log
set -u

export MUJOCO_GL=egl

echo "##### PHASE 0 ENVIRONMENT CHECKS #####"
echo "date        : $(date -Is)"
echo "host        : $(hostname)"
echo "conda env   : ${CONDA_DEFAULT_ENV:-<none>}"
echo "python      : $(python --version 2>&1)"
echo "MUJOCO_GL   : $MUJOCO_GL"
echo "lerobot src : $HOME/lerobot @ $(git -C "$HOME/lerobot" rev-parse HEAD 2>/dev/null)"
echo

echo "##### CHECKS 1-4 (driver / kernel module / torch+sm_120 / VRAM) #####"
python scripts/check_gpu.py
echo

echo "##### CHECK 5 (MuJoCo EGL render benchmark) #####"
python scripts/check_egl_fps.py
echo

echo "##### CHECK 6 (one LIBERO episode end to end) #####"
python scripts/check_libero_episode.py 2>&1 \
  | grep -vE 'Fetching [0-9]+ files|it/s\]|robosuite WARNING|WARNING:robosuite|UserWarning|warnings\.warn'
echo

echo "##### INSTALLED VERSIONS #####"
python - <<'PY'
import importlib.metadata as md
pkgs = ["lerobot", "torch", "torchvision", "torchcodec", "hf-libero", "robosuite",
        "robomimic", "bddl", "mujoco", "peft", "transformers", "gymnasium",
        "accelerate", "numpy", "wandb", "num2words", "hf-egl-probe", "egl_probe"]
for p in pkgs:
    try:
        print(f"{p:<14}= {md.version(p)}")
    except Exception:
        print(f"{p:<14}= NOT INSTALLED")
PY
echo "ffmpeg        = $(ffmpeg -version 2>/dev/null | head -1)"
echo
echo "##### END #####"
