#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ROOT="${LABWAM_DEPLOY_ROOT:-$(cd "$CODE_ROOT/../.." && pwd)}"
VARIANT="${1:-general}"
STEPS="${2:-10}"
case "$VARIANT" in general|take_tube|dispense|take_dropper|return_dropper|return_tube|water_bath) ;; *) echo 'Unknown model variant' >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TMPDIR="$DEPLOY_ROOT/.cache/tmp" XDG_CACHE_HOME="$DEPLOY_ROOT/.cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_EXTENSIONS_DIR="$DEPLOY_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$DEPLOY_ROOT/.cache/triton" CUDA_CACHE_PATH="$DEPLOY_ROOT/.cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$DEPLOY_ROOT/.cache/torchinductor"
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd "$CODE_ROOT"
WARMUP_RESET="${LABWAM_WARMUP_RESET:-}"
if [[ -z "$WARMUP_RESET" && -f "$DEPLOY_ROOT/runtime/warmup/$VARIANT.json" ]]; then
  WARMUP_RESET="$DEPLOY_ROOT/runtime/warmup/$VARIANT.json"
fi
# Enable compilation automatically only when startup can warm the actual model
# before declaring the HTTP service ready. A warmup never sends robot commands.
DEFAULT_COMPILE=none
if [[ -n "$WARMUP_RESET" ]]; then DEFAULT_COMPILE=default; fi
OPT_ARGS=(--compile-mode "${LABWAM_COMPILE_MODE:-$DEFAULT_COMPILE}")
if [[ "${LABWAM_CONDITION_CACHE:-1}" == 0 ]]; then OPT_ARGS+=(--no-condition-cache); fi
if [[ "${LABWAM_DIT_CACHE:-1}" != 0 ]]; then
  OPT_ARGS+=(--dit-cache-threshold "${LABWAM_DIT_CACHE_THRESHOLD:-0.99}" --dit-cache-max-skips "${LABWAM_DIT_CACHE_MAX_SKIPS:-3}")
fi
if [[ -n "$WARMUP_RESET" ]]; then OPT_ARGS+=(--warmup-reset "$WARMUP_RESET"); fi
exec "$DEPLOY_ROOT/.venv_inference/bin/python" -u scripts/serve_lab_policy.py \
  --checkpoint "$DEPLOY_ROOT/runtime/models/$VARIANT/checkpoint_step_30000.safetensors" \
  --steps "$STEPS" --bind "${LABWAM_BIND:-127.0.0.1}" --port "${LABWAM_PORT:-8849}" "${OPT_ARGS[@]}"
