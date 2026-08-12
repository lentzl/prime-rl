#!/usr/bin/env bash
set -euo pipefail

root=/home/ubuntu/prime-rl
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-base}
output_root=/ephemeral/subagent-rung/evals/271-276-qwen35-27b-mastery-battery-v1/${label}

configs=(
  271-qwen35-27b-mastery-foundations-baseline
  272-qwen35-27b-mastery-coordination-calibration
  273-qwen35-27b-mastery-coordination-heldout
  274-qwen35-27b-mastery-ownership-child-ood
  275-qwen35-27b-mastery-ownership-coordinator-ood
  276-qwen35-27b-mastery-oolong-ood
)

cd "$root"
mkdir -p "$output_root"
for name in "${configs[@]}"; do
  /home/ubuntu/.local/bin/uv run eval @ \
    "configs/debug/subagent-communication/${name}.toml" \
    --model "$model" \
    --output-dir "$output_root/$name"
done
