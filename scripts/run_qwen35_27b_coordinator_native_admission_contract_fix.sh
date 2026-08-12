#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
CONFIG=configs/debug/subagent-communication/270-qwen35-27b-coordinator-native-admission-contract-fix.toml
OUTPUT=/ephemeral/subagent-rung/evals/270-qwen35-27b-coordinator-native-admission-contract-fix-r1
INFERENCE_SERVICE=qwen35-27b-teacher-inference.service

cd "$ROOT"
export PATH="$ROOT/.venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
trap 'systemctl --user stop "$INFERENCE_SERVICE"' EXIT
rm -rf "$OUTPUT"

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

.venv/bin/eval @ "$CONFIG" --output-dir "$OUTPUT"
.venv/bin/python scripts/summarize_coordinator_native_admission.py \
  --traces "$OUTPUT/traces.jsonl" \
  --output "$OUTPUT/summary.json"
