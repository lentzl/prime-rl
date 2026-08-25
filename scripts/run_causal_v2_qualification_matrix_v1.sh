#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${CAUSAL_V2_OUTPUT_ROOT:-/home/ubuntu/rlm/results/causal-v2-qualification-v1}
axes=${CAUSAL_V2_PRINCIPAL_AXES:-natural_direct_control,natural_n1a,natural_n1a_local,natural_n1b}
num_tasks=${CAUSAL_V2_PRINCIPAL_DRAWS:-16}
q38_label=${CAUSAL_V2_Q38_LABEL:-q38-principal16-r1}
q35_label=${CAUSAL_V2_Q35_LABEL:-q35-principal16-r1}
q38_revision=${CAUSAL_V2_Q38_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
q35_revision=${CAUSAL_V2_Q35_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}
r7_model=${CAUSAL_V2_R7_MODEL:-}
r7_revision=${CAUSAL_V2_R7_REVISION:-}
r7_label=${CAUSAL_V2_R7_LABEL:-r7-principal16-r1}
require_r7=${CAUSAL_V2_REQUIRE_R7:-false}

case "$require_r7" in
  true|false) ;;
  *) echo "CAUSAL_V2_REQUIRE_R7 must be true or false" >&2; exit 1 ;;
esac
if [[ ! "$num_tasks" =~ ^[1-9][0-9]*$ ]]; then
  echo "CAUSAL_V2_PRINCIPAL_DRAWS must be a positive integer" >&2
  exit 1
fi

unset PROCEDURAL_NATURAL_YIELD_SCAFFOLD

run_leg() {
  local model=$1
  local label=$2
  local revision=$3
  local reasoning_effort=$4
  env \
    EVAL_MAX_NUM_BATCHED_TOKENS=512 \
    EVAL_MAX_NUM_SEQS=1 \
    EVAL_DISABLE_CUSTOM_ALL_REDUCE=true \
    QWEN38_QUALIFICATION_AXES="$axes" \
    QWEN38_QUALIFICATION_NUM_TASKS="$num_tasks" \
    QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
    QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
    QUALIFICATION_REASONING_EFFORT="$reasoning_effort" \
    PRIME_MASTERY_OUTPUT_ROOT="$output_root" \
    "$root/scripts/run_qwen38_27b_prime_harness_baseline_v1.sh" \
    "$model" "$label" "$revision"
}

run_leg Qwen/Qwen3.8-27B "$q38_label" "$q38_revision" xhigh
run_leg Qwen/Qwen3.5-27B "$q35_label" "$q35_revision" high

if [[ -n "$r7_model" && -n "$r7_revision" ]]; then
  run_leg "$r7_model" "$r7_label" "$r7_revision" high
elif [[ "$require_r7" == true ]]; then
  echo "R7 is required but CAUSAL_V2_R7_MODEL/CAUSAL_V2_R7_REVISION are unavailable" >&2
  exit 1
else
  echo "R7 leg skipped: protected snapshot or credentials unavailable on this host" >&2
fi
