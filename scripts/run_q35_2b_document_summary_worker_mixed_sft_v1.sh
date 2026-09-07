#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_trace=${1:?source summary-worker failure trace required}
source_model=${2:?source summary-worker checkpoint required}
run_name=${3:-h176-summary-worker-mixed-sft24-step2-v1}
optimizer_updates=${4:-2}
dataset_dir=${DOCUMENT_SUMMARY_MIXED_SFT_DATASET:-/home/ubuntu/rlm/data/q35-2b-document-summary-worker-mixed-sft-v1}
output_root=${DOCUMENT_SUMMARY_MIXED_SFT_OUTPUT_ROOT:-/home/ubuntu/rlm/outputs/q35-2b-document-summary-worker-mixed-sft-v1}
state_dir=${DOCUMENT_SUMMARY_MIXED_SFT_STATE_DIR:-/home/ubuntu/rlm/state/q35-2b-document-summary-worker-mixed-sft-v1}
python_bin=${PYTHON_BIN:-/home/ubuntu/rlm/prime-rl/.venv/bin/python3}

if [[ ! -x "$python_bin" ]]; then
  echo "pinned Python is unavailable: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$source_trace" || ! -f "$source_model/model.safetensors" || ! -f "$source_model/STABLE" ]]; then
  echo "mixed summary SFT source trace or checkpoint is incomplete" >&2
  exit 1
fi

cd "$root"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/home/ubuntu/rlm/prime-rl/.venv}
if [[ ! -e "$dataset_dir" ]]; then
  "$python_bin" scripts/export_q35_2b_document_summary_worker_mixed_sft_v1.py \
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
  --optimizer-updates "$optimizer_updates" \
  --timeout 3600
