#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_model=${1:?source summary checkpoint required}
scope_trace=${2:?scope scaffold trace required}
operations_trace=${3:?operations scaffold trace required}
exceptions_trace=${4:?exceptions scaffold trace required}
run_name=${5:-h176-summary-commit-revision-sft12-step1-v3}
optimizer_updates=${6:-1}
dataset_dir=${DOCUMENT_SUMMARY_COMMIT_REVISION_SFT_DATASET:-/home/ubuntu/rlm/data/q35-2b-document-summary-commit-revision-sft-v3}
output_root=${DOCUMENT_SUMMARY_COMMIT_REVISION_SFT_OUTPUT_ROOT:-/home/ubuntu/rlm/outputs/q35-2b-document-summary-commit-revision-sft-v3}
state_dir=${DOCUMENT_SUMMARY_COMMIT_REVISION_SFT_STATE_DIR:-/home/ubuntu/rlm/state/q35-2b-document-summary-commit-revision-sft-v3}
python_bin=${PYTHON_BIN:-/home/ubuntu/rlm/prime-rl/.venv/bin/python3}

if [[ ! -x "$python_bin" ]]; then
  echo "pinned Python is unavailable: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$source_model/model.safetensors" || ! -f "$source_model/STABLE" ]]; then
  echo "summary commit revision source checkpoint is incomplete" >&2
  exit 1
fi
for trace in "$scope_trace" "$operations_trace" "$exceptions_trace"; do
  if [[ ! -f "$trace" ]]; then
    echo "summary commit revision trace is missing: $trace" >&2
    exit 1
  fi
done

cd "$root"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/home/ubuntu/rlm/prime-rl/.venv}
export PYTHONPATH="$root/src:$root/scripts${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -e "$dataset_dir" ]]; then
  "$python_bin" scripts/export_q35_2b_document_summary_commit_revision_sft_v3.py \
    --traces "$scope_trace" \
    --traces "$operations_trace" \
    --traces "$exceptions_trace" \
    --output-dir "$dataset_dir"
fi
audit_path="$dataset_dir/RENDERER-AUDIT.json"
if [[ ! -e "$audit_path" ]]; then
  audit_tmp="$dataset_dir/.RENDERER-AUDIT.json.tmp.$$"
  trap 'rm -f "$audit_tmp"' EXIT
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$python_bin" scripts/audit_q35_2b_document_summary_commit_revision_renderer_v3.py \
      --traces "$scope_trace" \
      --traces "$operations_trace" \
      --traces "$exceptions_trace" \
      --tokenizer "$source_model" \
      --dataset-dir "$dataset_dir" >"$audit_tmp"
  mv "$audit_tmp" "$audit_path"
  trap - EXIT
fi
mkdir -p "$state_dir"
"$python_bin" scripts/run_q35_2b_document_decision_sft_v1.py \
  --repo "$root" \
  --source-model "$source_model" \
  --dataset-dir "$dataset_dir" \
  --output-root "$output_root" \
  --state-dir "$state_dir" \
  --run-name "$run_name" \
  --learning-rate 2e-7 \
  --optimizer-updates "$optimizer_updates" \
  --timeout 3600
