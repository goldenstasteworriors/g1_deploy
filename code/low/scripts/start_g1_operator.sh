#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ROOT="${LABWAM_DEPLOY_ROOT:-$(cd "$CODE_ROOT/../.." && pwd)}"
export TMPDIR="$DEPLOY_ROOT/.cache/tmp" ROS_LOG_DIR="$DEPLOY_ROOT/logs/ros"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$ROS_LOG_DIR"
exec "${LABWAM_CONTROLLER_PYTHON:-$DEPLOY_ROOT/.conda_controller/bin/python}" \
  "$CODE_ROOT/scripts/run_g1_operator.py" --sonic-root "$DEPLOY_ROOT/sonicmj" \
  --catalog "$DEPLOY_ROOT/assets/initial_poses/catalog.json" \
  --selection "$DEPLOY_ROOT/runtime/selected_initial_pose.json" \
  --marks "$DEPLOY_ROOT/runtime/marks.json" --logs "$DEPLOY_ROOT/logs/operator" \
  --camera-host "${LABWAM_CAMERA_HOST:-192.168.123.164}" --camera-port "${LABWAM_CAMERA_PORT:-5555}" \
  --interface "${LABWAM_G1_INTERFACE:-enp132s0}" --start-policy "$@"
