#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
owner_model=${1:?document-owner model path required}
worker_model=${2:?translation-worker model path required}
label=${3:-owner-e33-lineage-worker-h176-run1}
revision=${4:-existing-lineages-no-update}
output_root=${DOCUMENT_TRANSLATION_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-document-translation-prime-agent-v1}
receipt=$output_root/$label/SMOKE-RECEIPT.txt

if [[ -e "$receipt" ]]; then
  echo "refusing to overwrite document translation receipt: $receipt" >&2
  exit 1
fi

owner_sha_before=$(sha256sum "$owner_model/model.safetensors" | awk '{print $1}')
worker_sha_before=$(sha256sum "$worker_model/model.safetensors" | awk '{print $1}')

cd "$root"
DUAL_SCAFFOLD_PROFILE=custom \
DUAL_EXTERNAL_MODEL=q35-2b-document-translation-prime-agent-v1 \
DUAL_DEPTH_DEFAULT_CHILD=1 \
DOCUMENT_TRANSLATION_CONFIG="$root/experiments/qwen35-2b-document-translation-prime-agent-v1/smoke.toml" \
EVAL_DRIVER=scripts/run_q35_2b_document_translation_eval_v1.sh \
QWEN38_QUALIFICATION_OUTPUT_ROOT="$output_root" \
scripts/run_q35_2b_dual_policy_mastery_v1.sh \
  "$owner_model" "$worker_model" "$label" "$revision"

owner_sha_after=$(sha256sum "$owner_model/model.safetensors" | awk '{print $1}')
worker_sha_after=$(sha256sum "$worker_model/model.safetensors" | awk '{print $1}')
if [[ "$owner_sha_before" != "$owner_sha_after" || "$worker_sha_before" != "$worker_sha_after" ]]; then
  echo "model weights changed during no-update translation smoke" >&2
  exit 1
fi

mkdir -p "$(dirname "$receipt")"
{
  printf 'owner_model=%s\n' "$owner_model"
  printf 'owner_model_sha256=%s\n' "$owner_sha_after"
  printf 'worker_model=%s\n' "$worker_model"
  printf 'worker_model_sha256=%s\n' "$worker_sha_after"
  printf 'result=%s\n' "$output_root/$label/document"
} >"$receipt"

echo "document translation smoke completed: $receipt"
