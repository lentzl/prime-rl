#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/207-qwen35-4b-executable-step24-repeat-confirmation
CONFIG=configs/debug/subagent-communication/207-qwen35-4b-executable-step24-repeat-confirmation.toml
MODELS=(qwen35-4b-run100-step8 qwen35-4b-executable-step24)

cd "$ROOT"
rm -rf "$OUTPUT_ROOT"
for model in "${MODELS[@]}"; do
  /home/ubuntu/.local/bin/uv run eval @ "$CONFIG" \
    --model "$model" \
    --output-dir "$OUTPUT_ROOT/$model"
done
