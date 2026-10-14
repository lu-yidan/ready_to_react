#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE=disabled
python_bin="${READY_REACT_PYTHON:-.venv/bin/python}"
if [[ ! -d data/DuoBox/fbx_export_joints ]]; then
  echo 'Missing data/DuoBox/fbx_export_joints. Obtain DuoBox from the authors first.' >&2
  exit 1
fi
"$python_bin" reactmotion/scripts/main.py -t data_test -c configs/exps/vqvae_tokenizer.yaml dataloader_cfg.dataset_cfg.gen_motion_split=True
"$python_bin" reactmotion/scripts/main.py -t data_test -c configs/exps/vqvae_tokenizer.yaml dataloader_cfg.dataset_cfg.gen_preprocess=True
"$python_bin" reactmotion/scripts/main.py -t data_test -c configs/exps/reactive_model.yaml dataloader_cfg.dataset_cfg.gen_preprocess=True
"$python_bin" reactmotion/scripts/main.py -t data_test -c configs/exps/sparse_control.yaml dataloader_cfg.dataset_cfg.gen_motion_split=True
"$python_bin" reactmotion/scripts/main.py -t data_test -c configs/exps/sparse_control.yaml dataloader_cfg.dataset_cfg.gen_preprocess=True
