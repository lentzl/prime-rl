#!/usr/bin/env bash
set -euo pipefail

# Terminal-worker comparison: each GPU evaluates its own frozen checkpoint.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
label=${2:?paired evaluation label required}
baseline=${DOCUMENT_SUMMARY_BASELINE_MODEL:?baseline checkpoint required}
candidate=${DOCUMENT_SUMMARY_CANDIDATE_MODEL:?candidate checkpoint required}
output_root=${QWEN38_QUALIFICATION_OUTPUT_ROOT:?evaluation output root required}
receipt=$output_root/$label/PAIRED-TERMINAL-RECEIPT.txt
test ! -e "$receipt"

baseline_before=$(sha256sum "$baseline/model.safetensors" | awk '{print $1}')
candidate_before=$(sha256sum "$candidate/model.safetensors" | awk '{print $1}')

run_arm() {
  local arm=$1 model=$2 port=$3 probe
  for probe in exceptions city-shade; do
    DOCUMENT_SUMMARY_CONFIG="$root/experiments/qwen35-2b-document-summary-prime-agent-v1/smoke-evidence-$probe.toml" \
    EVAL_CLIENT_BASE_URL="http://127.0.0.1:$port/v1" \
      bash "$root/scripts/run_q35_2b_document_summary_eval_v1.sh" "$model" "$label-$arm-$probe"
  done
}

run_arm baseline "$baseline" "${COORDINATOR_BACKEND_PORT:-8101}" &
baseline_pid=$!
run_arm candidate "$candidate" "${CHILD_BACKEND_PORT:-8102}" &
candidate_pid=$!
status=0
wait "$baseline_pid" || status=1
wait "$candidate_pid" || status=1
test "$status" -eq 0

baseline_after=$(sha256sum "$baseline/model.safetensors" | awk '{print $1}')
candidate_after=$(sha256sum "$candidate/model.safetensors" | awk '{print $1}')
test "$baseline_before" = "$baseline_after"
test "$candidate_before" = "$candidate_after"
{
  printf 'mode=terminal-worker-native-prime-agent-direct-engines\n'
  printf 'role_proxy_used=false\n'
  printf 'baseline_model=%s\nbaseline_sha256=%s\n' "$baseline" "$baseline_after"
  printf 'candidate_model=%s\ncandidate_sha256=%s\n' "$candidate" "$candidate_after"
  printf 'episodes=4\nsemantic_review=separate\n'
} >"$receipt"
