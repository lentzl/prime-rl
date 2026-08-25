#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.8-27B}
label=${2:?a composite-workload label is required}
revision=${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
parent_output_root=${PRIME_MASTERY_OUTPUT_ROOT:?PRIME_MASTERY_OUTPUT_ROOT is required}
workload_root=$parent_output_root/$label/workload
driver=$root/scripts/run_qwen38_27b_prime_harness_qualification_v1.sh

if [[ -e "$workload_root" ]]; then
  echo "refusing to overwrite composite workload output: $workload_root" >&2
  exit 1
fi

env -u QWEN38_QUALIFICATION_START_INDEX \
  QWEN38_QUALIFICATION_OUTPUT_ROOT="$workload_root" \
  QWEN38_QUALIFICATION_AXES=natural_direct_control,natural_n1a,natural_n1a_local,natural_n1b \
  QWEN38_QUALIFICATION_NUM_TASKS=1 \
  QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
  QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
  QWEN38_QUALIFICATION_INDEX_OFFSET=11 \
  QUALIFICATION_REASONING_EFFORT=xhigh \
  MODEL_REVISION="$revision" \
  "$driver" "$model" mixed-current-causal

env -u QWEN38_QUALIFICATION_INDEX_OFFSET \
  QWEN38_QUALIFICATION_OUTPUT_ROOT="$workload_root" \
  QWEN38_QUALIFICATION_AXES=natural_n1a_local \
  QWEN38_QUALIFICATION_NUM_TASKS=5 \
  QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
  QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
  QWEN38_QUALIFICATION_START_INDEX=3806011 \
  QUALIFICATION_REASONING_EFFORT=xhigh \
  MODEL_REVISION="$revision" \
  "$driver" "$model" n1a-local-recovery-sized
