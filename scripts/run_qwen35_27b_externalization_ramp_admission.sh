#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-base}
output_root=${PRIME_EXTERNALIZATION_OUTPUT_ROOT:-/ephemeral/subagent-rung/evals/332-333-qwen35-27b-externalization-ramp-admission}/${label}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
client_base_url=${EVAL_CLIENT_BASE_URL:-http://127.0.0.1:8501/v1}

cd "$root"
for name in \
  332-qwen35-27b-oolong-labeled-admission \
  333-qwen35-27b-oolong-recursive-admission
do
  "$eval_bin" @ "configs/debug/subagent-communication/${name}.toml" \
    --model "$model" \
    --output-dir "$output_root/$name" \
    --client.base-url "$client_base_url"
done
