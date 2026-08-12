#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/202-qwen35-4b-executable-causal
CONFIG=configs/debug/subagent-communication/202-qwen35-4b-executable-causal-screen.toml

declare -A ADAPTERS=(
  [qwen35-4b-run100-step8]="/ephemeral/subagent-rung/outputs/100-qwen35-4b-single-admission-sft-r1/weights/step_8"
  [qwen35-4b-executable-step8]="/ephemeral/subagent-rung/outputs/201-qwen35-4b-executable-causal-sft-r1/weights/step_8"
  [qwen35-4b-executable-step16]="/ephemeral/subagent-rung/outputs/201-qwen35-4b-executable-causal-sft-r1/weights/step_16"
  [qwen35-4b-executable-step24]="/ephemeral/subagent-rung/outputs/201-qwen35-4b-executable-causal-sft-r1/weights/step_24"
  [qwen35-4b-executable-step32]="/ephemeral/subagent-rung/outputs/201-qwen35-4b-executable-causal-sft-r1/weights/step_32"
  [qwen35-4b-executable-step40]="/ephemeral/subagent-rung/outputs/201-qwen35-4b-executable-causal-sft-r1/weights/step_40"
)

MODELS=(
  qwen35-4b-run100-step8
  qwen35-4b-executable-step8
  qwen35-4b-executable-step16
  qwen35-4b-executable-step24
  qwen35-4b-executable-step32
  qwen35-4b-executable-step40
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
