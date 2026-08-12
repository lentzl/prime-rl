#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/222-qwen35-27b-bidirectional-selection-r2
CONFIG=configs/debug/subagent-communication/222-qwen35-27b-bidirectional-selection-screen.toml

declare -A ADAPTERS=(
  [qwen35-27b-harness-step4]="/ephemeral/subagent-rung/outputs/217-qwen35-27b-harness-mastery-sft-r1/weights/step_4"
  [qwen35-27b-bidirectional-step4]="/ephemeral/subagent-rung/outputs/221-qwen35-27b-bidirectional-sft-r1/weights/step_4"
  [qwen35-27b-bidirectional-step8]="/ephemeral/subagent-rung/outputs/221-qwen35-27b-bidirectional-sft-r1/weights/step_8"
  [qwen35-27b-bidirectional-step12]="/ephemeral/subagent-rung/outputs/221-qwen35-27b-bidirectional-sft-r1/weights/step_12"
)

MODELS=(
  qwen35-27b-harness-step4
  qwen35-27b-bidirectional-step4
  qwen35-27b-bidirectional-step8
  qwen35-27b-bidirectional-step12
)

cd "$ROOT"
for model in "${MODELS[@]}"; do
  if ! curl --fail --silent http://127.0.0.1:8000/v1/models | grep --quiet "$model"; then
    curl --fail --silent --show-error \
      --request POST http://127.0.0.1:8100/v1/load_lora_adapter \
      --header 'Content-Type: application/json' \
      --data "{\"lora_name\":\"$model\",\"lora_path\":\"${ADAPTERS[$model]}\"}"
    printf '\n'
  fi
done

mkdir -p "$OUTPUT_ROOT"
for model in "${MODELS[@]}"; do
  /home/ubuntu/.local/bin/uv run eval @ "$CONFIG" \
    --model "$model" \
    --output-dir "$OUTPUT_ROOT/$model"
done
