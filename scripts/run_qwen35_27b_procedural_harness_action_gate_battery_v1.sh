#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-candidate}
start_index=${HARNESS_ACTION_GATE_START_INDEX:-1200000}
admission_driver=$root/scripts/run_qwen35_27b_procedural_harness_action_admission_v1.sh
inference_driver=$root/scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh
client_base_url=${EVAL_CLIENT_BASE_URL:-}
requested_rungs=${HARNESS_ACTION_GATE_RUNGS:-atomic_state,atomic_send,atomic_followup}

cd "$root"
if [[ ! -x "$admission_driver" ]]; then
  echo "harness-action admission driver is not executable: $admission_driver" >&2
  exit 1
fi
if [[ ! "$start_index" =~ ^[0-9]+$ ]]; then
  echo "harness-action gate start index must be non-negative: $start_index" >&2
  exit 1
fi

if [[ -z "$client_base_url" ]]; then
  evaluation_root=${PROCEDURAL_HARNESS_OUTPUT_ROOT:-${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-procedural-harness-action-ramp-v1}}
  PRIME_MASTERY_OUTPUT_ROOT=$evaluation_root \
  EVAL_DRIVER=$root/scripts/run_qwen35_27b_procedural_harness_action_gate_battery_v1.sh \
  EVAL_EXPERIMENT_DIR=experiments/qwen35-27b-procedural-harness-master-v1 \
    exec "$inference_driver" "$model" "$label" "${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}"
fi

rungs=(atomic_state atomic_send atomic_followup atomic_child_request)
IFS=, read -ra selected_rungs <<<"$requested_rungs"
if [[ ${#selected_rungs[@]} -eq 0 ]]; then
  echo "HARNESS_ACTION_GATE_RUNGS must select at least one rung" >&2
  exit 1
fi
for rung in "${selected_rungs[@]}"; do
  offset=-1
  for index in "${!rungs[@]}"; do
    if [[ "${rungs[$index]}" == "$rung" ]]; then
      offset=$index
      break
    fi
  done
  if ((offset < 0)); then
    echo "unknown harness-action gate rung: $rung" >&2
    exit 1
  fi
  gate_start=$((start_index + offset * 1000))
  HARNESS_ACTION_ADMISSION_START_INDEX=$gate_start \
    "$admission_driver" "$rung" "$model" "$label-$rung-gate-r1"
done

echo "harness-action gate battery completed: $label"
