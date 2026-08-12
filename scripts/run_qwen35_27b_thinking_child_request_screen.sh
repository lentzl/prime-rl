#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/227-qwen35-27b-thinking-child-request
CONFIG=configs/debug/subagent-communication/227-qwen35-27b-thinking-child-request-screen.toml

declare -A ADAPTERS=(
  [qwen35-27b-thinking-request-step1]="/ephemeral/subagent-rung/outputs/226-qwen35-27b-thinking-child-request-opsd-r1/weights/step_1"
  [qwen35-27b-thinking-request-step2]="/ephemeral/subagent-rung/outputs/226-qwen35-27b-thinking-child-request-opsd-r1/weights/step_2"
  [qwen35-27b-thinking-request-step3]="/ephemeral/subagent-rung/outputs/226-qwen35-27b-thinking-child-request-opsd-r1/weights/step_3"
  [qwen35-27b-thinking-request-step4]="/ephemeral/subagent-rung/outputs/226-qwen35-27b-thinking-child-request-opsd-r1/weights/step_4"
)

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

/home/ubuntu/.local/bin/uv run eval @ "$CONFIG" \
  --model Qwen/Qwen3.5-27B \
  --output-dir "$OUTPUT_ROOT/base"

for model in \
  qwen35-27b-thinking-request-step1 \
  qwen35-27b-thinking-request-step2 \
  qwen35-27b-thinking-request-step3 \
  qwen35-27b-thinking-request-step4; do
  if ! curl --fail --silent http://127.0.0.1:8000/v1/models | grep --quiet "$model"; then
    curl --fail --silent --show-error \
      --request POST http://127.0.0.1:8100/v1/load_lora_adapter \
      --header 'Content-Type: application/json' \
      --data "{\"lora_name\":\"$model\",\"lora_path\":\"${ADAPTERS[$model]}\"}"
    printf '\n'
  fi
  /home/ubuntu/.local/bin/uv run eval @ "$CONFIG" \
    --model "$model" \
    --output-dir "$OUTPUT_ROOT/${model##*-step}"
done
