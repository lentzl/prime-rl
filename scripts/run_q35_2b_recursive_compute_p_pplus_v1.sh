#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
child_model=${2:?child model path required}
revision=${3:-c54-h176-fixed}
experiment=$root/experiments/qwen35-2b-document-recursion-zero-update-v1
output_root=${RECURSIVE_COMPUTE_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-recursive-compute-p-pplus-v1}
uv_bin=${UV_BIN:-$(command -v uv || true)}

if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin=$HOME/.local/bin/uv
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv executable not found" >&2
  exit 1
fi

p_config=$experiment/recursive-compute-p-v1.toml
pplus_config=$experiment/recursive-compute-pplus-v1.toml
for path in "$coordinator_model/STABLE" "$coordinator_model/model.safetensors" \
  "$child_model/STABLE" "$child_model/model.safetensors" "$p_config" "$pplus_config"; do
  if [[ ! -f "$path" ]]; then
    echo "required recursive-compute input is missing: $path" >&2
    exit 1
  fi
done

cd "$root"
coordinator_sha_before=$(sha256sum "$coordinator_model/model.safetensors" | awk '{print $1}')
child_sha_before=$(sha256sum "$child_model/model.safetensors" | awk '{print $1}')

run_arm() {
  local config=$1
  local label=$2
  DUAL_SCAFFOLD_PROFILE=custom \
  DUAL_EXTERNAL_MODEL=q35-2b-document-recursion-zero-update \
  DUAL_LEAK_DOCUMENT_MANAGER_EXACT_ACTION=1 \
  DUAL_ROOT_COORDINATOR_CONTRACT=1 \
  DUAL_DOCUMENT_ROOT_CAUSAL_UTILITY_DECISION_CONTRACT=1 \
  DUAL_DOCUMENT_LEAF_COMPUTE_REPORT_SCAFFOLD=1 \
  DUAL_DOCUMENT_MANAGER_FANIN_SCAFFOLD=1 \
  DUAL_DOCUMENT_ROOT_TOPOLOGY_NORMALIZATION_SCAFFOLD=1 \
  DUAL_DOCUMENT_ROOT_TYPED_TOPOLOGY_DECISION=1 \
  DUAL_DOCUMENT_ROOT_FLAT_FANIN_SCAFFOLD=1 \
  DUAL_DEPTH_DEFAULT_CHILD=1 \
  DOCUMENT_RECURSION_CONFIG="$config" \
  EVAL_DRIVER=scripts/run_q35_2b_document_recursion_eval_v1.sh \
  QWEN38_QUALIFICATION_OUTPUT_ROOT="$output_root" \
  UV_BIN="$uv_bin" \
  scripts/run_q35_2b_dual_policy_mastery_v1.sh \
    "$coordinator_model" "$child_model" "$label" "$revision"
}

run_arm "$p_config" p-depth2-attempt1-v1
run_arm "$pplus_config" pplus-depth3-attempt3-v1

coordinator_sha_after=$(sha256sum "$coordinator_model/model.safetensors" | awk '{print $1}')
child_sha_after=$(sha256sum "$child_model/model.safetensors" | awk '{print $1}')
if [[ "$coordinator_sha_before" != "$coordinator_sha_after" ]]; then
  echo "coordinator weights changed during P/P+ evaluation" >&2
  exit 1
fi
if [[ "$child_sha_before" != "$child_sha_after" ]]; then
  echo "child weights changed during P/P+ evaluation" >&2
  exit 1
fi

echo "recursive-compute P/P+ evaluations completed without weight changes: $output_root"
