#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:?model path is required}
label=${2:?gate label is required}
evaluation_root=${PROCEDURAL_HARNESS_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-natural-yield-sdpo-v1/low-dose-r1-gates}
natural_start=${NATURAL_YIELD_GATE_START_INDEX:-3500000}
local_work_start=${NATURAL_YIELD_LOCAL_WORK_GATE_START_INDEX:-3500001}
state_start=${NATURAL_YIELD_STATE_GATE_START_INDEX:-3510000}
send_start=${NATURAL_YIELD_SEND_GATE_START_INDEX:-3511000}

cd "$root"
if [[ -z "${EVAL_CLIENT_BASE_URL:-}" ]]; then
  echo "natural-yield gate battery requires an existing inference endpoint" >&2
  exit 1
fi

NATURAL_POLICY_RUNG=natural_n1 \
NATURAL_POLICY_PROBE_START_INDEX=$natural_start \
PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  "$root/scripts/run_qwen35_27b_natural_policy_connectivity_probe_v1.sh" \
  "$model" "$label-natural-yield"

NATURAL_POLICY_RUNG=natural_n1 \
NATURAL_POLICY_PROBE_START_INDEX=$local_work_start \
PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  "$root/scripts/run_qwen35_27b_natural_policy_connectivity_probe_v1.sh" \
  "$model" "$label-natural-yield-local-work"

HARNESS_ACTION_ADMISSION_START_INDEX=$state_start \
PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  "$root/scripts/run_qwen35_27b_procedural_harness_action_admission_v1.sh" \
  atomic_state "$model" "$label-atomic-state"

HARNESS_ACTION_ADMISSION_START_INDEX=$send_start \
PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  "$root/scripts/run_qwen35_27b_procedural_harness_action_admission_v1.sh" \
  atomic_send "$model" "$label-atomic-send"

echo "natural-yield gate battery completed: $label"
