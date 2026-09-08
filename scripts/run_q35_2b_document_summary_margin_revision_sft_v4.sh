#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_model=${1:?source summary checkpoint required}
scope_trace=${2:?scope margin-scaffold trace required}
operations_trace=${3:?operations margin-scaffold trace required}
exceptions_trace=${4:?exceptions margin-scaffold trace required}
run_name=${5:-h176-summary-margin-revision-sft12-step4-v4}
optimizer_updates=${6:-4}
learning_rate=${7:-2e-7}
dataset_dir=${DOCUMENT_SUMMARY_MARGIN_REVISION_SFT_DATASET:-/home/ubuntu/rlm/data/q35-2b-document-summary-margin-revision-sft-v4}
output_root=${DOCUMENT_SUMMARY_MARGIN_REVISION_SFT_OUTPUT_ROOT:-/home/ubuntu/rlm/outputs/q35-2b-document-summary-margin-revision-sft-v4}
state_dir=${DOCUMENT_SUMMARY_MARGIN_REVISION_SFT_STATE_DIR:-/home/ubuntu/rlm/state/q35-2b-document-summary-margin-revision-sft-v4}
python_bin=${PYTHON_BIN:-/home/ubuntu/rlm/prime-rl/.venv/bin/python3}

if [[ ! -x "$python_bin" ]]; then
  echo "pinned Python is unavailable: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$source_model/model.safetensors" || ! -f "$source_model/STABLE" ]]; then
  echo "summary margin revision source checkpoint is incomplete" >&2
  exit 1
fi
for trace in "$scope_trace" "$operations_trace" "$exceptions_trace"; do
  if [[ ! -f "$trace" ]]; then
    echo "summary margin revision trace is missing: $trace" >&2
    exit 1
  fi
done

cd "$root"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/home/ubuntu/rlm/prime-rl/.venv}
export PYTHONPATH="$root/src:$root/scripts${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -e "$dataset_dir" ]]; then
  "$python_bin" scripts/export_q35_2b_document_summary_margin_revision_sft_v4.py \
    --traces "$scope_trace" \
    --traces "$operations_trace" \
    --traces "$exceptions_trace" \
    --source-model "$source_model" \
    --output-dir "$dataset_dir"
fi
audit_path="$dataset_dir/RENDERER-AUDIT.json"
if [[ ! -e "$audit_path" ]]; then
  audit_tmp="$dataset_dir/.RENDERER-AUDIT.json.tmp.$$"
  trap 'rm -f "$audit_tmp"' EXIT
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$python_bin" scripts/audit_q35_2b_document_summary_margin_revision_renderer_v4.py \
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
  --learning-rate "$learning_rate" \
  --optimizer-updates "$optimizer_updates" \
  --timeout 3600
