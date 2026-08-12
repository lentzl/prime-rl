#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/218-219-qwen35-27b-harness-mastery
STANDARD=configs/debug/subagent-communication/218-qwen35-27b-harness-mastery-standard-screen.toml
CLEAN=configs/debug/subagent-communication/219-qwen35-27b-harness-mastery-clean-screen.toml

declare -A ADAPTERS=(
  [qwen35-27b-harness-step4]="/ephemeral/subagent-rung/outputs/217-qwen35-27b-harness-mastery-sft-r1/weights/step_4"
  [qwen35-27b-harness-step12]="/ephemeral/subagent-rung/outputs/217-qwen35-27b-harness-mastery-sft-r1/weights/step_12"
  [qwen35-27b-harness-step24]="/ephemeral/subagent-rung/outputs/217-qwen35-27b-harness-mastery-sft-r1/weights/step_24"
)

MODELS=(
  qwen35-27b-harness-step4
  qwen35-27b-harness-step12
  qwen35-27b-harness-step24
)

cd "$ROOT"
for model in "${MODELS[@]}"; do
  curl --fail --silent --show-error \
    --request POST http://127.0.0.1:8100/v1/load_lora_adapter \
    --header 'Content-Type: application/json' \
    --data "{\"lora_name\":\"$model\",\"lora_path\":\"${ADAPTERS[$model]}\"}"
  printf '\n'
done

mkdir -p "$OUTPUT_ROOT"
/home/ubuntu/.local/bin/uv run eval @ "$STANDARD" \
  --model Qwen/Qwen3.5-27B \
  --output-dir "$OUTPUT_ROOT/base-standard"
/home/ubuntu/.local/bin/uv run eval @ "$CLEAN" \
  --model Qwen/Qwen3.5-27B \
  --output-dir "$OUTPUT_ROOT/base-clean"
for model in "${MODELS[@]}"; do
  /home/ubuntu/.local/bin/uv run eval @ "$STANDARD" \
    --model "$model" \
    --output-dir "$OUTPUT_ROOT/${model}-standard"
  /home/ubuntu/.local/bin/uv run eval @ "$CLEAN" \
    --model "$model" \
    --output-dir "$OUTPUT_ROOT/${model}-clean"
done
