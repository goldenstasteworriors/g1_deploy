#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ROOT="${LABWAM_DEPLOY_ROOT:-$(cd "$CODE_ROOT/../.." && pwd)}"
BASE_PYTHON="${LABWAM_BASE_PYTHON:-/home/yanjun/anaconda3/envs/wam/bin/python}"
UV="${LABWAM_UV:-/home/yanjun/.local/bin/uv}"
export TMPDIR="$DEPLOY_ROOT/.cache/tmp" UV_CACHE_DIR="$DEPLOY_ROOT/.cache/uv"
export PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL=''
mkdir -p "$TMPDIR"
"$BASE_PYTHON" -c 'import sys,torch; assert sys.version_info[:2]==(3,11); assert torch.__version__.split("+")[0]=="2.5.1"; print("Base checked:",torch.__version__)'
if [[ ! -x "$DEPLOY_ROOT/.venv_inference/bin/python" ]]; then
  "$UV" venv --system-site-packages --python "$BASE_PYTHON" "$DEPLOY_ROOT/.venv_inference"
fi
# --no-deps preserves the existing torch/CUDA stack. All additions stay here.
"$UV" pip install --python "$DEPLOY_ROOT/.venv_inference/bin/python" --no-deps \
  --index-url https://pypi.org/simple -r "$CODE_ROOT/configs/requirements-g1-inference.txt"
cd "$CODE_ROOT"
"$DEPLOY_ROOT/.venv_inference/bin/python" -c 'from lab_low.deployment import InferenceSession; from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir; print("INFERENCE_IMPORTS_OK")'
