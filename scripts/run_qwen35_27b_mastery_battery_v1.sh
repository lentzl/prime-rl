#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-base}
output_root=${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/subagent-rung/evals/271-276-qwen35-27b-mastery-battery-v1}/${label}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}

configs=(
  271-qwen35-27b-mastery-foundations-baseline
  272-qwen35-27b-mastery-coordination-calibration
  273-qwen35-27b-mastery-coordination-heldout
  274-qwen35-27b-mastery-ownership-child-ood
  275-qwen35-27b-mastery-ownership-coordinator-ood
  276-qwen35-27b-mastery-oolong-ood
)

cd "$root"
if [[ ! -x "$eval_bin" ]]; then
  echo "eval executable not found: $eval_bin" >&2
  exit 1
fi
mkdir -p "$output_root"
for name in "${configs[@]}"; do
  "$eval_bin" @ \
    "configs/debug/subagent-communication/${name}.toml" \
    --model "$model" \
    --output-dir "$output_root/$name"
done
