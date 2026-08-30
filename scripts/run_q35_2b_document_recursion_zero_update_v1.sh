#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
child_model=${2:?child model path required}
label=${3:-latent-document-v1}
revision=${4:-candidate-local}
output_root=${DOCUMENT_RECURSION_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-document-recursion-zero-update-v1}
experiment=$root/experiments/qwen35-2b-document-recursion-zero-update-v1
receipt=$output_root/$label-RECEIPT.json
uv_bin=${UV_BIN:-$(command -v uv || true)}

if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin=$HOME/.local/bin/uv
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv executable not found" >&2
  exit 1
fi
if [[ -e "$receipt" ]]; then
  echo "refusing to overwrite document recursion receipt: $receipt" >&2
  exit 1
fi

cd "$root"
coordinator_sha_before=$(sha256sum "$coordinator_model/model.safetensors" | awk '{print $1}')
child_sha_before=$(sha256sum "$child_model/model.safetensors" | awk '{print $1}')

for topology in direct flat hierarchical; do
  topology_label=$label-$topology
  DUAL_SCAFFOLD_PROFILE=custom \
  DUAL_EXTERNAL_MODEL=q35-2b-document-recursion-zero-update \
  DUAL_DEPTH_DEFAULT_CHILD=1 \
  DOCUMENT_RECURSION_CONFIG="$experiment/$topology.toml" \
  EVAL_DRIVER=scripts/run_q35_2b_document_recursion_eval_v1.sh \
  QWEN38_QUALIFICATION_OUTPUT_ROOT="$output_root" \
  UV_BIN="$uv_bin" \
  scripts/run_q35_2b_dual_policy_mastery_v1.sh \
    "$coordinator_model" "$child_model" "$topology_label" "$revision"
done

coordinator_sha_after=$(sha256sum "$coordinator_model/model.safetensors" | awk '{print $1}')
child_sha_after=$(sha256sum "$child_model/model.safetensors" | awk '{print $1}')
if [[ "$coordinator_sha_before" != "$coordinator_sha_after" ]]; then
  echo "coordinator weights changed during zero-update evaluation" >&2
  exit 1
fi
if [[ "$child_sha_before" != "$child_sha_after" ]]; then
  echo "child weights changed during zero-update evaluation" >&2
  exit 1
fi

"$uv_bin" run --no-sync scripts/summarize_q35_2b_document_recursion_v1.py \
  --output-root "$output_root" \
  --label "$label" \
  --coordinator-sha256 "$coordinator_sha_after" \
  --child-sha256 "$child_sha_after" \
  --output "$receipt"

echo "document recursion zero-update evaluation completed: $receipt"
