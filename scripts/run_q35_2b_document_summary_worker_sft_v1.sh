#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_trace=${1:?source summary-worker trace required}
source_model=${2:?source H176 checkpoint required}
run_name=${3:-h176-summary-worker-sft12-step2-v1}
dataset_dir=${DOCUMENT_SUMMARY_SFT_DATASET:-/home/ubuntu/rlm/data/q35-2b-document-summary-worker-sft-v1}
output_root=${DOCUMENT_SUMMARY_SFT_OUTPUT_ROOT:-/home/ubuntu/rlm/outputs/q35-2b-document-summary-worker-sft-v1}
state_dir=${DOCUMENT_SUMMARY_SFT_STATE_DIR:-/home/ubuntu/rlm/state/q35-2b-document-summary-worker-sft-v1}
python_bin=${PYTHON_BIN:-/home/ubuntu/rlm/prime-rl/.venv/bin/python3}

if [[ ! -x "$python_bin" ]]; then
  echo "pinned Python is unavailable: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$source_trace" || ! -f "$source_model/model.safetensors" || ! -f "$source_model/STABLE" ]]; then
  echo "summary SFT source trace or checkpoint is incomplete" >&2
  exit 1
fi

cd "$root"
if [[ ! -e "$dataset_dir" ]]; then
  "$python_bin" scripts/export_q35_2b_document_summary_worker_sft_v1.py \
    --traces "$source_trace" \
    --output-dir "$dataset_dir"
fi
mkdir -p "$state_dir"
"$python_bin" scripts/run_q35_2b_document_decision_sft_v1.py \
  --repo "$root" \
  --source-model "$source_model" \
  --dataset-dir "$dataset_dir" \
  --output-root "$output_root" \
  --state-dir "$state_dir" \
  --run-name "$run_name" \
  --learning-rate 1e-6 \
  --optimizer-updates 2 \
  --timeout 3600
