#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.9 .venv
fi
uv pip install --python .venv/bin/python \
  'torch==1.13.0+cu116' 'torchvision==0.14.0+cu116' 'torchaudio==0.13.0+cu116' \
  --index-url https://download.pytorch.org/whl/cu116
uv pip install --python .venv/bin/python \
  'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl' \
  -r requirements-runtime.lock -c requirements-compat.txt \
  'torch==1.13.0+cu116' 'torchvision==0.14.0+cu116' 'torchaudio==0.13.0+cu116' \
  iopath fvcore wheel 'setuptools<81' --index-url https://pypi.org/simple
uv pip install --python .venv/bin/python -e . --no-build-isolation --no-deps
uv pip check --python .venv/bin/python
