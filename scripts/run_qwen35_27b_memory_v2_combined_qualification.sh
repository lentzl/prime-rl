#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:?usage: run_qwen35_27b_memory_v2_combined_qualification.sh MODEL LABEL}
label=${2:?usage: run_qwen35_27b_memory_v2_combined_qualification.sh MODEL LABEL}
output_root=${PRIME_MASTERY_OUTPUT_ROOT:?PRIME_MASTERY_OUTPUT_ROOT is required}
run_output=$output_root/$label
memory_root=$run_output/memory
mastery_root=$run_output/mastery

cd "$root"
if [[ -e "$run_output/QUALIFICATION_COMPLETE" ]]; then
  echo "qualification already completed: $run_output" >&2
  exit 1
fi

PRIME_MASTERY_OUTPUT_ROOT=$memory_root \
  scripts/run_qwen35_27b_memory_v2_full_eval.sh "$model" results
.venv/bin/python scripts/summarize_programmatic_memory_eval.py \
  "$memory_root/results" \
  --output "$run_output/memory-summary.json"

PRIME_MASTERY_OUTPUT_ROOT=$mastery_root \
  scripts/run_qwen35_27b_mastery_battery_v1.sh "$model" results
mastery_traces=()
while IFS= read -r path; do
  mastery_traces+=("$path")
done < <(find "$mastery_root/results" -type f -name traces.jsonl -print | sort)
if (( ${#mastery_traces[@]} == 0 )); then
  echo "mastery battery produced no traces: $mastery_root/results" >&2
  exit 1
fi
.venv/bin/python scripts/summarize_prime_agent_mastery.py \
  --json --summary-only "${mastery_traces[@]}" \
  >"$run_output/mastery-summary.json"

printf '%s\n' "model=$model" "label=$label" >"$run_output/QUALIFICATION_COMPLETE"
echo "combined qualification completed: $run_output"
