#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-candidate}
start_index=${HARNESS_ACTION_GATE_START_INDEX:-1200000}
admission_driver=$root/scripts/run_qwen35_27b_procedural_harness_action_admission_v1.sh

cd "$root"
if [[ ! -x "$admission_driver" ]]; then
  echo "harness-action admission driver is not executable: $admission_driver" >&2
  exit 1
fi
if [[ ! "$start_index" =~ ^[0-9]+$ ]]; then
  echo "harness-action gate start index must be non-negative: $start_index" >&2
  exit 1
fi

rungs=(atomic_state atomic_send atomic_followup)
for offset in "${!rungs[@]}"; do
  rung=${rungs[$offset]}
  gate_start=$((start_index + offset * 1000))
  HARNESS_ACTION_ADMISSION_START_INDEX=$gate_start \
    "$admission_driver" "$rung" "$model" "$label-$rung-gate-r1"
done

echo "harness-action gate battery completed: $label"
