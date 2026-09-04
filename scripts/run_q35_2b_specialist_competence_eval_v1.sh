#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
generic_worker_model=${2:?generic worker model path required}
specialist_model=${3:?specialist model path required}
expert_id=${4:?expert id required}
label=${5:?evaluation label required}
revision=${6:?model revision required}
config=${7:?frozen evaluation config path required}
output_root=${SPECIALIST_COMPETENCE_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-specialist-competence-s1-v1}
force_fixed_action=${SPECIALIST_COMPETENCE_FORCE_FIXED_ACTION:-0}

if [[ "$force_fixed_action" != 0 && "$force_fixed_action" != 1 ]]; then
  echo "SPECIALIST_COMPETENCE_FORCE_FIXED_ACTION must be 0 or 1" >&2
  exit 1
fi

case "$expert_id" in
  table_analyst)
    table_analyst_model=$specialist_model
    source_inspector_model=$generic_worker_model
    ;;
  source_inspector)
    table_analyst_model=$generic_worker_model
    source_inspector_model=$specialist_model
    ;;
  *)
    echo "expert id must be table_analyst or source_inspector" >&2
    exit 1
    ;;
esac

models=("$coordinator_model" "$generic_worker_model" "$specialist_model")
before=()
for model in "${models[@]}"; do
  if [[ "$model" != /* || ! -f "$model/STABLE" || ! -f "$model/model.safetensors" ]]; then
    echo "competence evaluation model is not an absolute stable checkpoint: $model" >&2
    exit 1
  fi
  before+=("$(sha256sum "$model/model.safetensors" | awk '{print $1}')")
done

DUAL_SCAFFOLD_PROFILE=custom \
DUAL_EXTERNAL_MODEL=q35-2b-specialist-competence \
DUAL_ROOT_COORDINATOR_CONTRACT=1 \
DUAL_SPECIALIST_WORKER_ROUTING=1 \
DUAL_SPECIALIST_FIXED_EXPERT="$expert_id" \
DUAL_SPECIALIST_FORCE_FIXED_ACTION="$force_fixed_action" \
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
    echo "model weights changed during specialist competence evaluation: ${models[$index]}" >&2
    exit 1
  fi
done

echo "fixed-route specialist competence evaluation completed: $output_root/$label"
