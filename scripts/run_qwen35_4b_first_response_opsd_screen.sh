#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/210-qwen35-4b-first-response-opsd
CONFIG=configs/debug/subagent-communication/210-qwen35-4b-first-response-opsd-screen.toml

declare -A ADAPTERS=(
  [qwen35-4b-run100-step8]="/ephemeral/subagent-rung/outputs/100-qwen35-4b-single-admission-sft-r1/weights/step_8"
  [qwen35-4b-first-response-step1]="/ephemeral/subagent-rung/outputs/209-qwen35-4b-first-response-opsd-r2/weights/step_1"
  [qwen35-4b-first-response-step2]="/ephemeral/subagent-rung/outputs/209-qwen35-4b-first-response-opsd-r2/weights/step_2"
  [qwen35-4b-first-response-step3]="/ephemeral/subagent-rung/outputs/209-qwen35-4b-first-response-opsd-r2/weights/step_3"
  [qwen35-4b-first-response-step4]="/ephemeral/subagent-rung/outputs/209-qwen35-4b-first-response-opsd-r2/weights/step_4"
)

MODELS=(
  qwen35-4b-run100-step8
  qwen35-4b-first-response-step1
  qwen35-4b-first-response-step2
  qwen35-4b-first-response-step3
  qwen35-4b-first-response-step4
)

cd "$ROOT"
for model in "${MODELS[@]}"; do
  curl --fail --silent --show-error \
    --request POST http://127.0.0.1:8100/v1/load_lora_adapter \
    --header 'Content-Type: application/json' \
    --data "{\"lora_name\":\"$model\",\"lora_path\":\"${ADAPTERS[$model]}\"}"
  printf '\n'
done

rm -rf "$OUTPUT_ROOT"
for model in "${MODELS[@]}"; do
  /home/ubuntu/.local/bin/uv run eval @ "$CONFIG" \
    --model "$model" \
    --output-dir "$OUTPUT_ROOT/$model"
done
