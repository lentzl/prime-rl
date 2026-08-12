#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/233-qwen35-27b-natural-control-selection-r1
CONFIG=configs/debug/subagent-communication/233-qwen35-27b-natural-control-selection.toml
ADAPTER_NAME=qwen35-27b-natural-control-step1
ADAPTER_PATH=/ephemeral/subagent-rung/outputs/232-qwen35-27b-natural-control-grpo-dose-r1/weights/step_1

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

/home/ubuntu/.local/bin/uv run eval @ "$CONFIG" \
  --model Qwen/Qwen3.5-27B \
  --output-dir "$OUTPUT_ROOT/base"

if ! curl --fail --silent http://127.0.0.1:8000/v1/models | grep --quiet "$ADAPTER_NAME"; then
  curl --fail --silent --show-error \
    --request POST http://127.0.0.1:8100/v1/load_lora_adapter \
    --header 'Content-Type: application/json' \
    --data "{\"lora_name\":\"$ADAPTER_NAME\",\"lora_path\":\"$ADAPTER_PATH\"}"
  printf '\n'
fi

/home/ubuntu/.local/bin/uv run eval @ "$CONFIG" \
  --model "$ADAPTER_NAME" \
  --output-dir "$OUTPUT_ROOT/candidate"
