#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_trace=${1:?source summary text trace required}
source_model=${2:?source summary checkpoint required}
run_name=${3:-h176-summary-live-revision-sft12-step1-v2}
optimizer_updates=${4:-1}
dataset_dir=${DOCUMENT_SUMMARY_LIVE_REVISION_SFT_DATASET:-/home/ubuntu/rlm/data/q35-2b-document-summary-live-revision-sft-v2}
output_root=${DOCUMENT_SUMMARY_LIVE_REVISION_SFT_OUTPUT_ROOT:-/home/ubuntu/rlm/outputs/q35-2b-document-summary-live-revision-sft-v2}
state_dir=${DOCUMENT_SUMMARY_LIVE_REVISION_SFT_STATE_DIR:-/home/ubuntu/rlm/state/q35-2b-document-summary-live-revision-sft-v2}
python_bin=${PYTHON_BIN:-/home/ubuntu/rlm/prime-rl/.venv/bin/python3}

if [[ ! -x "$python_bin" ]]; then
  echo "pinned Python is unavailable: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$source_trace" || ! -f "$source_model/model.safetensors" || ! -f "$source_model/STABLE" ]]; then
  echo "summary live revision source trace or checkpoint is incomplete" >&2
  exit 1
fi

cd "$root"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/home/ubuntu/rlm/prime-rl/.venv}
export PYTHONPATH="$root/src:$root/scripts${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -e "$dataset_dir" ]]; then
  "$python_bin" scripts/export_q35_2b_document_summary_live_revision_sft_v2.py \
    --traces "$source_trace" \
    --output-dir "$dataset_dir"
fi
audit_path="$dataset_dir/RENDERER-AUDIT.json"
if [[ ! -e "$audit_path" ]]; then
  audit_tmp="$dataset_dir/.RENDERER-AUDIT.json.tmp.$$"
  trap 'rm -f "$audit_tmp"' EXIT
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$python_bin" scripts/audit_q35_2b_document_summary_live_revision_renderer_v2.py \
      --trace "$source_trace" \
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
  --enable-thinking \
  --timeout 3600
