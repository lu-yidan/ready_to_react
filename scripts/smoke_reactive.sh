#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE=disabled
python_bin="${READY_REACT_PYTHON:-.venv/bin/python}"
"$python_bin" reactmotion/scripts/main.py -t test -c configs/exps/reactive_model.yaml \
  model_cfg.inference_func=reactive \
  val_dataloader_cfg.max_iter=1 val_dataloader_cfg.num_workers=0 \
  runner_cfg.evaluator_cfg.feature_compute=False \
  runner_cfg.evaluator_cfg.apd_compute=False \
  runner_cfg.evaluator_cfg.vis_output=1 "$@"
