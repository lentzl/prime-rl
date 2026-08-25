#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen38-to-qwen35-2b-role-distillation-v1
role=${1:-}
student_snapshot=${2:-}
role_dataset=${3:-}
baseline_manifest=${4:-}
expected_student_sha=c75915dd41cd4fc9b1a1ef5582c6fd14913fc6f9971a58feca3b72b4bfcad406

case "$role" in
  orchestrator|child) ;;
  *) echo "usage: $0 {orchestrator|child} STUDENT_SNAPSHOT ROLE_DATASET BASELINE_MANIFEST" >&2; exit 1 ;;
esac
if [[ -z "$student_snapshot" || -z "$role_dataset" || -z "$baseline_manifest" ]]; then
  echo "usage: $0 {orchestrator|child} STUDENT_SNAPSHOT ROLE_DATASET BASELINE_MANIFEST" >&2
  exit 1
fi
template=$experiment/$role-sft-lora.toml
if [[ ! -f "$template" ]]; then
  echo "missing role SFT config: $template" >&2
  exit 1
fi
if [[ ! -f "$student_snapshot/STABLE" || ! -f "$student_snapshot/model.safetensors" ]]; then
  echo "student snapshot is not the expected stable dense export: $student_snapshot" >&2
  exit 1
fi
actual_student_sha=$(sha256sum "$student_snapshot/model.safetensors" | awk '{print $1}')
if [[ "$actual_student_sha" != "$expected_student_sha" ]]; then
  echo "student weight hash mismatch: $actual_student_sha" >&2
  exit 1
fi
if [[ ! -f "$role_dataset/train.parquet" || ! -f "$role_dataset/MANIFEST.json" ]]; then
  echo "role dataset is incomplete: $role_dataset" >&2
  exit 1
fi
if [[ ! -f "$baseline_manifest" ]]; then
  echo "untouched-student baseline manifest is missing: $baseline_manifest" >&2
  exit 1
fi

cd "$root"
export PATH="$HOME/.local/bin:$root/.venv/bin:$PATH"
validation_json=$(uv run --frozen --no-sync python \
  scripts/validate_q35_2b_role_training_inputs_v1.py \
  --role "$role" \
  --role-dataset "$role_dataset" \
  --baseline-manifest "$baseline_manifest" \
  --student-snapshot "$student_snapshot" \
  --template "$template")
printf 'validated role-SFT inputs: %s\n' "$validation_json"

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch role SFT while another GPU process is active" >&2
  exit 1
fi

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
command=(
  uv run --frozen --no-sync sft @ "$template"
  --model.name "$student_snapshot"
  --tokenizer.name "$student_snapshot"
  --data.name "$role_dataset"
)
if [[ "${ROLE_SFT_DRY_RUN:-false}" == true ]]; then
  command+=(--dry-run)
fi
"${command[@]}"
