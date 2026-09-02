#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
generic_worker_model=${2:?generic worker model path required}
table_analyst_model=${3:?table analyst model path required}
source_inspector_model=${4:?source inspector model path required}
label=${5:?evaluation label required}
revision=${6:-specialist-worker-population-c1-v1}
config=${7:?evaluation config path required}
output_root=${SPECIALIST_POPULATION_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-specialist-population-c1-v1}

models=(
  "$coordinator_model"
  "$generic_worker_model"
  "$table_analyst_model"
  "$source_inspector_model"
)
before=()
for model in "${models[@]}"; do
  before+=("$(sha256sum "$model/model.safetensors" | awk '{print $1}')")
done

DUAL_SCAFFOLD_PROFILE=custom \
DUAL_EXTERNAL_MODEL=q35-2b-specialist-population \
DUAL_ROOT_COORDINATOR_CONTRACT=1 \
DUAL_SPECIALIST_WORKER_ROUTING=1 \
DUAL_TABLE_ANALYST_MODEL="$table_analyst_model" \
DUAL_SOURCE_INSPECTOR_MODEL="$source_inspector_model" \
DUAL_DEPTH_DEFAULT_CHILD=1 \
DOCUMENT_RECURSION_CONFIG="$config" \
EVAL_DRIVER=scripts/run_q35_2b_document_recursion_eval_v1.sh \
QWEN38_QUALIFICATION_OUTPUT_ROOT="$output_root" \
"$root/scripts/run_q35_2b_dual_policy_mastery_v1.sh" \
  "$coordinator_model" "$generic_worker_model" "$label" "$revision"

for index in "${!models[@]}"; do
  after=$(sha256sum "${models[$index]}/model.safetensors" | awk '{print $1}')
  if [[ "${before[$index]}" != "$after" ]]; then
    echo "model weights changed during specialist population evaluation: ${models[$index]}" >&2
    exit 1
  fi
done

echo "specialist population evaluation completed without weight changes: $output_root/$label"
