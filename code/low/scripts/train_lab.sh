#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
OPENWAM_ENV="${OPENWAM_ENV:-/root/ykj/OpenWAM/.conda-envs/openwam}"
export PATH="$OPENWAM_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/.cache/torchinductor"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "The lab launcher requires exactly one visible GPU." >&2
  exit 2
fi
FREE_MIB="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)"
if (( FREE_MIB < ${LAB_MIN_FREE_MIB:-76000} )); then
  echo "GPU $CUDA_VISIBLE_DEVICES has only $FREE_MIB MiB free; choose an idle 80 GB GPU for this foundation configuration." >&2
  exit 2
fi
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
exec "$OPENWAM_ENV/bin/torchrun" --nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29642}" scripts/train_lab.py \
  "project.wandb.run_name=$(date +%Y-%m-%d)_lab_low" "$@"
