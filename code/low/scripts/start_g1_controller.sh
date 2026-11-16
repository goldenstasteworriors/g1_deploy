#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ROOT="${LABWAM_DEPLOY_ROOT:-$(cd "$CODE_ROOT/../.." && pwd)}"
export TMPDIR="$DEPLOY_ROOT/.cache/tmp" ROS_LOG_DIR="$DEPLOY_ROOT/logs/ros"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$ROS_LOG_DIR"
exec "${LABWAM_CONTROLLER_PYTHON:-$DEPLOY_ROOT/.conda_controller/bin/python}" \
  "$CODE_ROOT/scripts/run_g1_lab_controller.py" --sonic-root "$DEPLOY_ROOT/sonicmj" "$@"
