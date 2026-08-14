#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-base}
output_root=${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/subagent-rung/evals/347-348-qwen35-27b-memory-v2-full}/${label}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
client_base_url=${EVAL_CLIENT_BASE_URL:-}

configs=(
  347-qwen35-27b-memory-v2-familiar-full-eval
  348-qwen35-27b-memory-v2-ood-full-eval
)

cd "$root"
if [[ ! -x "$eval_bin" ]]; then
  echo "eval executable not found: $eval_bin" >&2
  exit 1
fi
for split in familiar_heldout semantic_ood; do
  path="/ephemeral/subagent-rung/data/programmatic-episodic-memory-v2/${split}.jsonl"
  if [[ ! -f "$path" ]]; then
    echo "frozen memory split not found: $path" >&2
    exit 1
  fi
done

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
