#!/usr/bin/env bash
# Task-local connection policy; does not modify ~/.ssh/config.
# OpenSSH retries connection establishment, not remote commands that ran and failed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTROL="$ROOT/.cache/ai2ykj-control"
PROXY='ssh -o ConnectTimeout=180 -o ConnectionAttempts=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o IPQoS=none -o KexAlgorithms=curve25519-sha256 -W %h:%p ai2ykj'
if ssh -S "$CONTROL" -O check -- ai2ykj >/dev/null 2>&1; then
  PROXY="ssh -S $CONTROL -W %h:%p ai2ykj"
fi
exec ssh \
  -o ControlMaster=auto -o "ControlPath=$ROOT/.cache/a800-control" \
  -o "ProxyCommand=$PROXY" -o IPQoS=none -o KexAlgorithms=curve25519-sha256 \
  -o ConnectTimeout=180 -o ConnectionAttempts=10 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=10 \
  -- A800-1 "$@"
