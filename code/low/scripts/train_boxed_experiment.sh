#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
EXPERIMENT="${1:?general|take_tube|dispense|take_dropper}"
shift
case "$EXPERIMENT" in general|take_tube|dispense|take_dropper) ;; *) exit 2;; esac
if [[ -f "outputs/boxed_v3_30k/parallel/$EXPERIMENT.json" && "${LAB_PARALLEL_OWNER:-0}" != 1 ]]; then
  exec "${OPENWAM_ENV:-/share/personal/yekangjie/lab-low/envs/openwam}/bin/python" scripts/run_boxed_parallel.py "$EXPERIMENT" --wait
fi
ASSETS="${LAB_ASSETS:-/share/personal/yekangjie/lab-low/assets}"
FOUNDATION="$ASSETS/OpenWAM-Alpha-Pretrain-Foundation-Model"
export OPENWAM_ENV="${OPENWAM_ENV:-/share/personal/yekangjie/lab-low/envs/openwam}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MASTER_PORT="${MASTER_PORT:-29648}"
exec bash scripts/train_lab.sh \
  "model.architecture.foundation_checkpoint=$FOUNDATION/checkpoint_step_154000.safetensors" \
  "training.finetune_ckpt_path=$FOUNDATION" \
  "dataloader.manifest=data/boxed_v3_suite/$EXPERIMENT.json" \
  "dataloader.artifact_dir=.cache/boxed_v3_$EXPERIMENT" \
  training.max_steps=30000 training.save_steps=1000 \
  validation_every=1000 validation_windows=10 \
  "training.output_path=outputs/boxed_v3_30k/$EXPERIMENT" \
  "project.wandb.run_name=$(date +%Y-%m-%d)_boxed_v3_$EXPERIMENT" "$@"
