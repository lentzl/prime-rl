#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${CAUSAL_V2_OUTPUT_ROOT:-/home/ubuntu/rlm/results/causal-v2-qualification-v1}
q38_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
q35_revision=fc05daec18b0a78c049392ed2e771dde82bdf654

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing recovery while another GPU process is active" >&2
  exit 1
fi

run_leg() {
  local model=$1
  local label=$2
  local revision=$3
  local axes=$4
  local draws=$5
  local reasoning_effort=$6
  local start_index=${7:-}

  env \
    EVAL_MAX_NUM_BATCHED_TOKENS=512 \
    EVAL_MAX_NUM_SEQS=1 \
    EVAL_DISABLE_CUSTOM_ALL_REDUCE=true \
    QWEN38_QUALIFICATION_AXES="$axes" \
    QWEN38_QUALIFICATION_NUM_TASKS="$draws" \
    QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
    QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
    QWEN38_QUALIFICATION_START_INDEX="$start_index" \
    QUALIFICATION_REASONING_EFFORT="$reasoning_effort" \
    PRIME_MASTERY_OUTPUT_ROOT="$output_root" \
    "$root/scripts/run_qwen38_27b_prime_harness_baseline_v1.sh" \
    "$model" "$label" "$revision"
}

# The original Q38 block has a valid, durable prefix for local tasks 0-9.
# Recover only the provider-error suffix, preserving exact task identities.
run_leg \
  Qwen/Qwen3.8-27B \
  q38-principal16-r3-local-cont-3806010-6-r1 \
  "$q38_revision" \
  natural_n1a_local \
  6 \
  xhigh \
  3806010

run_leg \
  Qwen/Qwen3.8-27B \
  q38-principal16-r3-n1b-r1 \
  "$q38_revision" \
  natural_n1b \
  16 \
  xhigh

run_leg \
  Qwen/Qwen3.5-27B \
  q35-principal16-r3 \
  "$q35_revision" \
  natural_direct_control,natural_n1a,natural_n1a_local,natural_n1b \
  16 \
  high
