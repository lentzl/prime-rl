#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
INFERENCE_SERVICE=qwen35-27b-teacher-inference.service
CONFIG=configs/debug/subagent-communication/245-qwen35-27b-first-ownership-grpo-dose.toml

cd "$ROOT"
systemctl --user restart "$INFERENCE_SERVICE"
for _ in $(seq 1 150); do
  if curl --fail --silent http://127.0.0.1:8100/health >/dev/null \
    && curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:8100/health >/dev/null
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null

export HF_TOKEN="$HF_KEY"
exec /home/ubuntu/prime-rl/.venv/bin/rl @ "$CONFIG"
