#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-base}
output_root=${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/subagent-rung/evals/320-326-qwen35-27b-mastery-fast-screen-v1}/${label}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
client_base_url=${EVAL_CLIENT_BASE_URL:-}

configs=(
  320-qwen35-27b-mastery-fast-foundations
  321-qwen35-27b-mastery-fast-coordination
  322-qwen35-27b-mastery-fast-ownership-child
  323-qwen35-27b-mastery-fast-ownership-coordinator
  324-qwen35-27b-mastery-fast-ownership-child-xml
  325-qwen35-27b-mastery-fast-ownership-coordinator-xml
  326-qwen35-27b-mastery-fast-oolong
)

cd "$root"
if [[ ! -x "$eval_bin" ]]; then
  echo "eval executable not found: $eval_bin" >&2
  exit 1
fi
mkdir -p "$output_root"
for name in "${configs[@]}"; do
  args=(
    "$eval_bin" @
    "configs/debug/subagent-communication/${name}.toml"
    --model "$model"
    --output-dir "$output_root/$name"
  )
  if [[ -n "$client_base_url" ]]; then
    args+=(--client.base-url "$client_base_url")
  fi
  "${args[@]}"
done
