#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
CONFIG=configs/debug/subagent-communication/250-qwen35-27b-guided-native-ownership-collection-r2.toml
OUTPUT=/ephemeral/subagent-rung/evals/250-qwen35-27b-guided-native-ownership-collection-r2

cd "$ROOT"
rm -rf "$OUTPUT"
if ! curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then
  systemctl --user restart qwen35-27b-teacher-inference.service
fi
for _ in $(seq 1 150); do
  if curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null

exec /home/ubuntu/prime-rl/.venv/bin/eval @ "$CONFIG"
