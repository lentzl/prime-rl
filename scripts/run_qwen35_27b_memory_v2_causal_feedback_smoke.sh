#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-base}
output_root=${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/subagent-rung/evals/342-qwen35-27b-memory-v2-causal-feedback-smoke}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
client_base_url=${EVAL_CLIENT_BASE_URL:-}

cd "$root"
args=(
  "$eval_bin" @
  configs/debug/subagent-communication/342-qwen35-27b-memory-v2-causal-feedback-smoke.toml
  --model "$model"
  --output-dir "$output_root/$label/342-qwen35-27b-memory-v2-causal-feedback-smoke"
)
if [[ -n "$client_base_url" ]]; then
  args+=(--client.base-url "$client_base_url")
fi
"${args[@]}"
