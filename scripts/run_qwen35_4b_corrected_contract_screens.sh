#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/200-qwen35-4b-corrected-contract
CONFIG=configs/debug/subagent-communication/200-qwen35-4b-corrected-contract-screen.toml

declare -A ADAPTERS=(
  [qwen35-4b-run100-step8]="/ephemeral/subagent-rung/outputs/100-qwen35-4b-single-admission-sft-r1/weights/step_8"
  [qwen35-4b-contract-step2]="/ephemeral/subagent-rung/outputs/199-qwen35-4b-corrected-contract-sft-r1/weights/step_2"
  [qwen35-4b-contract-step4]="/ephemeral/subagent-rung/outputs/199-qwen35-4b-corrected-contract-sft-r1/weights/step_4"
  [qwen35-4b-contract-step6]="/ephemeral/subagent-rung/outputs/199-qwen35-4b-corrected-contract-sft-r1/weights/step_6"
  [qwen35-4b-contract-step8]="/ephemeral/subagent-rung/outputs/199-qwen35-4b-corrected-contract-sft-r1/weights/step_8"
)

MODELS=(
  qwen35-4b-run100-step8
  qwen35-4b-contract-step2
  qwen35-4b-contract-step4
  qwen35-4b-contract-step6
  qwen35-4b-contract-step8
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
