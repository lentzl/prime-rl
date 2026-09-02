#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
child_model=${2:?child model path required}
revision=${3:-c54-h176-same-depth-search-v1}
experiment=$root/experiments/qwen35-2b-document-recursion-zero-update-v1
output_root=${SAME_DEPTH_SEARCH_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-same-depth-search-p-pplus-v1}
state_path=${SAME_DEPTH_SEARCH_STATE_PATH:-/home/ubuntu/rlm/state/q35-2b-same-depth-search-p-pplus-v1.json}
p_config=${SAME_DEPTH_SEARCH_P_CONFIG:-$experiment/recursive-same-depth-search-p-v1.toml}
pplus_config=${SAME_DEPTH_SEARCH_PPLUS_CONFIG:-$experiment/recursive-same-depth-search-pplus-v1.toml}
p_label=${SAME_DEPTH_SEARCH_P_LABEL:-p-depth2-attempt1-v1}
pplus_label=${SAME_DEPTH_SEARCH_PPLUS_LABEL:-pplus-depth2-attempt4-v1}
pplus_attempts=${SAME_DEPTH_SEARCH_PPLUS_ATTEMPTS:-4}
expected_tasks=${SAME_DEPTH_SEARCH_EXPECTED_TASKS:-12}
uv_bin=${UV_BIN:-$(command -v uv || true)}

if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin=$HOME/.local/bin/uv
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv executable not found" >&2
  exit 1
fi

for path in "$coordinator_model/STABLE" "$coordinator_model/model.safetensors" \
  "$child_model/STABLE" "$child_model/model.safetensors" "$p_config" "$pplus_config"; do
  if [[ ! -f "$path" ]]; then
    echo "required same-depth search input is missing: $path" >&2
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
  DUAL_ROOT_COORDINATOR_CONTRACT=1 \
  DUAL_DOCUMENT_ROOT_CAUSAL_UTILITY_DECISION_CONTRACT=1 \
  DUAL_DOCUMENT_LEAF_COMPUTE_REPORT_SCAFFOLD=1 \
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

run_arm "$p_config" "$p_label"
run_arm "$pplus_config" "$pplus_label"

coordinator_sha_after=$(sha256sum "$coordinator_model/model.safetensors" | awk '{print $1}')
child_sha_after=$(sha256sum "$child_model/model.safetensors" | awk '{print $1}')
if [[ "$coordinator_sha_before" != "$coordinator_sha_after" ]]; then
  echo "coordinator weights changed during same-depth P/P+ evaluation" >&2
  exit 1
fi
if [[ "$child_sha_before" != "$child_sha_after" ]]; then
  echo "child weights changed during same-depth P/P+ evaluation" >&2
  exit 1
fi

mkdir -p "$(dirname "$state_path")"
"$uv_bin" run --no-sync scripts/summarize_q35_2b_recursive_compute_pair_v1.py \
  --p "$output_root/$p_label/document/document/traces.jsonl" \
  --p-plus "$output_root/$pplus_label/document/document/traces.jsonl" \
  --expected-tasks "$expected_tasks" \
  --p-plus-attempts "$pplus_attempts" \
  --gap-floor 4 \
  --output "$state_path"

echo "same-depth search P/P+ evaluation completed without weight changes: $output_root"
