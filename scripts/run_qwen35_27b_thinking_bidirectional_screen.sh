#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/225-qwen35-27b-thinking-bidirectional
CONFIG=configs/debug/subagent-communication/225-qwen35-27b-thinking-bidirectional-screen.toml
MODEL=qwen35-27b-harness-step12
ADAPTER=/ephemeral/subagent-rung/outputs/217-qwen35-27b-harness-mastery-sft-r1/weights/step_12

cd "$ROOT"
if ! curl --fail --silent http://127.0.0.1:8000/v1/models | grep --quiet "$MODEL"; then
  curl --fail --silent --show-error \
    --request POST http://127.0.0.1:8100/v1/load_lora_adapter \
    --header 'Content-Type: application/json' \
    --data "{\"lora_name\":\"$MODEL\",\"lora_path\":\"$ADAPTER\"}"
  printf '\n'
fi

mkdir -p "$OUTPUT_ROOT"
/home/ubuntu/.local/bin/uv run eval @ "$CONFIG" \
  --model Qwen/Qwen3.5-27B \
  --output-dir "$OUTPUT_ROOT/base"
/home/ubuntu/.local/bin/uv run eval @ "$CONFIG" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_ROOT/step12"
