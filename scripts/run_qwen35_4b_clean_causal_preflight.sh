#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/208-qwen35-4b-clean-causal-preflight
CONFIG=configs/debug/subagent-communication/208-qwen35-4b-clean-causal-preflight.toml

cd "$ROOT"
rm -rf "$OUTPUT_ROOT"
/home/ubuntu/.local/bin/uv run eval @ "$CONFIG" --output-dir "$OUTPUT_ROOT"
