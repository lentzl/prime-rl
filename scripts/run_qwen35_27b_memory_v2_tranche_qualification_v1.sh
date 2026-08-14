#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_output=${MEMORY_TRANCHE_OUTPUT:-/ephemeral/subagent-rung/outputs/346-qwen35-27b-memory-v2-hybrid-tranche-v2}
output_root=${MEMORY_QUALIFICATION_OUTPUT:-/ephemeral/subagent-rung/evals/349-qwen35-27b-memory-v2-tranche-qualification-v1}
base_revision=fc05daec18b0a78c049392ed2e771dde82bdf654
steps=(1 2 4 8)

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to qualify while another GPU process is active" >&2
  exit 1
fi
for step in "${steps[@]}"; do
  checkpoint=$source_output/weights/step_$step
  if [[ ! -f "$checkpoint/STABLE" ]]; then
    echo "selected checkpoint is not stable: $checkpoint" >&2
    exit 1
  fi
done

models=(
  Qwen/Qwen3.5-27B
  "$source_output/weights/step_1"
  "$source_output/weights/step_2"
  "$source_output/weights/step_4"
  "$source_output/weights/step_8"
)
labels=(base step-1 step-2 step-4 step-8)
revisions=("$base_revision" "" "" "" "")

for index in "${!models[@]}"; do
  run_output=$output_root/${labels[$index]}
  if [[ -f "$run_output/QUALIFICATION_COMPLETE" ]]; then
    echo "skipping completed qualification: ${labels[$index]}"
    continue
  fi
  if [[ -e "$run_output" ]]; then
    echo "refusing to mix partial qualification output: $run_output" >&2
    exit 1
  fi
  PRIME_MASTERY_OUTPUT_ROOT=$output_root \
    MASTERY_EVAL_DRIVER=scripts/run_qwen35_27b_memory_v2_combined_qualification.sh \
    scripts/run_qwen35_27b_mastery_fast_screen_model_v1.sh \
      "${models[$index]}" "${labels[$index]}" "${revisions[$index]}"
done

echo "tranche qualification completed: $output_root"
