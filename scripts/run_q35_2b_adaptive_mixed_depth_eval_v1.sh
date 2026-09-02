#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
child_model=${2:?child model path required}
label=${3:?evaluation label required}
revision=${4:-c54-h176-adaptive-mixed-depth-v1}
experiment=$root/experiments/qwen35-2b-document-recursion-zero-update-v1
config=${ADAPTIVE_MIXED_DEPTH_CONFIG:-$experiment/adaptive-mixed-depth-smoke-v1.toml}
output_root=${ADAPTIVE_MIXED_DEPTH_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-adaptive-mixed-depth-v1}

coordinator_sha_before=$(sha256sum "$coordinator_model/model.safetensors" | awk '{print $1}')
child_sha_before=$(sha256sum "$child_model/model.safetensors" | awk '{print $1}')

DUAL_SCAFFOLD_PROFILE=custom \
DUAL_EXTERNAL_MODEL=q35-2b-adaptive-mixed-depth \
DUAL_ROOT_COORDINATOR_CONTRACT=1 \
DUAL_ADAPTIVE_DOCUMENT_DECISION=1 \
DUAL_DOCUMENT_LEAF_COMPUTE_REPORT_SCAFFOLD=1 \
DUAL_DOCUMENT_MANAGER_FANIN_SCAFFOLD=1 \
DUAL_DOCUMENT_MANAGER_WAIT_SCAFFOLD=1 \
DUAL_DOCUMENT_MANAGER_TERMINATION_SCAFFOLD=1 \
DUAL_DOCUMENT_ROOT_REPORT_RELAY_SCAFFOLD=1 \
DUAL_DOCUMENT_ROOT_TOPOLOGY_NORMALIZATION_SCAFFOLD=1 \
DUAL_DOCUMENT_ROOT_FLAT_FANIN_SCAFFOLD=1 \
DUAL_DEPTH_DEFAULT_CHILD=1 \
DOCUMENT_RECURSION_CONFIG="$config" \
EVAL_DRIVER=scripts/run_q35_2b_document_recursion_eval_v1.sh \
QWEN38_QUALIFICATION_OUTPUT_ROOT="$output_root" \
scripts/run_q35_2b_dual_policy_mastery_v1.sh \
  "$coordinator_model" "$child_model" "$label" "$revision"

coordinator_sha_after=$(sha256sum "$coordinator_model/model.safetensors" | awk '{print $1}')
child_sha_after=$(sha256sum "$child_model/model.safetensors" | awk '{print $1}')
if [[ "$coordinator_sha_before" != "$coordinator_sha_after" ]]; then
  echo "coordinator weights changed during adaptive mixed-depth evaluation" >&2
  exit 1
fi
if [[ "$child_sha_before" != "$child_sha_after" ]]; then
  echo "child weights changed during adaptive mixed-depth evaluation" >&2
  exit 1
fi

echo "adaptive mixed-depth evaluation completed without weight changes: $output_root/$label"
