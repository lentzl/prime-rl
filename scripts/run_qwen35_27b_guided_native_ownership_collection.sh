#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
CONFIG=configs/debug/subagent-communication/249-qwen35-27b-guided-native-ownership-collection.toml
OUTPUT=/ephemeral/subagent-rung/evals/249-qwen35-27b-guided-native-ownership-collection-r1

cd "$ROOT"
rm -rf "$OUTPUT"
systemctl --user restart qwen35-27b-teacher-inference.service
for _ in $(seq 1 150); do
  if curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null

exec /home/ubuntu/prime-rl/.venv/bin/eval @ "$CONFIG"
