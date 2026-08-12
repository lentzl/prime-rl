#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/244-qwen35-27b-thinking-ownership-selection-r1
CONFIG=configs/debug/subagent-communication/244-qwen35-27b-thinking-ownership-selection.toml
ADAPTER_NAME=qwen35-27b-thinking-ownership-step1
ADAPTER_PATH=/ephemeral/subagent-rung/outputs/243-qwen35-27b-thinking-ownership-bootstrap-sft-r1/weights/step_1

cd "$ROOT"
rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

systemctl --user restart qwen35-27b-teacher-inference.service
for _ in $(seq 1 150); do
  if curl --fail --silent http://127.0.0.1:8100/health >/dev/null \
    && curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:8100/health >/dev/null
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null

/home/ubuntu/prime-rl/.venv/bin/eval @ "$CONFIG" \
  --model Qwen/Qwen3.5-27B \
  --output-dir "$OUTPUT_ROOT/base"

curl --fail --silent --show-error \
  --request POST http://127.0.0.1:8100/v1/load_lora_adapter \
  --header 'Content-Type: application/json' \
  --data "{\"lora_name\":\"$ADAPTER_NAME\",\"lora_path\":\"$ADAPTER_PATH\"}"
printf '\n'

/home/ubuntu/prime-rl/.venv/bin/eval @ "$CONFIG" \
  --model "$ADAPTER_NAME" \
  --output-dir "$OUTPUT_ROOT/candidate"
