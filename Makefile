# Teach-by-Demonstration VLA Cell - teaching cost curve
#
# Everything runs inside the `lerobot` conda environment with MUJOCO_GL=egl.
# See docs/00-environment.md for how that environment is built.

SHELL := /bin/bash
CONDA := source $$HOME/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
export MUJOCO_GL := egl

DATASET_REV := a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
DATASET_ROOT := $$HOME/.cache/huggingface/hub/datasets--lerobot--libero/snapshots/$(DATASET_REV)
STAGE1 := checkpoints/stage1/checkpoints/last/pretrained_model

.PHONY: help verify data splits sweep stage1 train curve clean-results reproduce

help:
	@echo "  verify         Phase 0: run the six environment checks"
	@echo "  data           Phase 2: inventory the dataset (downloads it on first run)"
	@echo "  splits         Phase 2: regenerate the K-shot splits (deterministic)"
	@echo "  sweep          Phase 3: measure what batch size fits, and how fast"
	@echo "  stage1         Phase 3: pre-train the factory's existing line"
	@echo "  train          Phase 3: run the full K-shot study (long; resumable)"
	@echo "  curve          Phase 4: draw the teaching cost curve from results/"
	@echo "  clean-results  Remove derived outputs, keeping checkpoints"
	@echo "  reproduce      Everything, end to end"

verify:  ## Phase 0: run the six environment checks
	@$(CONDA) && bash scripts/run_phase0_checks.sh

data:  ## Phase 2: inventory the dataset (downloads it on first run)
	@$(CONDA) && python src/data/inspect_dataset.py --dataset-root "$(DATASET_ROOT)"

splits:  ## Phase 2: regenerate the K-shot splits (deterministic)
	@$(CONDA) && python src/data/make_kshot_split.py

sweep:  ## Phase 3: measure what batch size fits, and how fast
	@$(CONDA) && python src/train/batch_size_sweep.py --scratch /tmp/bs_sweep
	@$(CONDA) && python src/train/budget.py

stage1:  ## Phase 3: pre-train the "factory's existing line"
	@$(CONDA) && python src/train/run_train.py --split stage1 --steps 6000 --batch-size 16

train:  ## Phase 3: run the full K-shot study (long; idempotent, safe to resume)
	@$(CONDA) && bash scripts/run_phase3.sh 6000 10 6000

curve:  ## Phase 4: draw the teaching cost curve from results/
	@$(CONDA) && python src/analysis/plot_curve.py --baseline results/stage1_baseline

clean-results:  ## Remove derived outputs, keeping checkpoints
	rm -rf results/teaching_cost_curve.json assets/teaching_cost_curve.png

# The full study takes many hours on one 8 GB GPU; see docs/03-training.md for
# the measured budget. Each stage is idempotent, so an interrupted run resumes.
reproduce: verify data splits train curve  ## Everything, end to end
	@echo "Done. Main figure: assets/teaching_cost_curve.png"
