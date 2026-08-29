#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
child_model=${2:?child model path required}
label=${3:?evaluation label required}
start_index=${4:?start index required}
num_tasks=${5:-6}
revision=${6:-candidate-local}
artifact_root=${Q35_2B_TIGHT_REPORT_ARTIFACT_ROOT:-/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1}
output_root=${Q35_2B_TIGHT_REPORT_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1}
bootstrap=$artifact_root/$label-bootstrap.json

if [[ ! "$start_index" =~ ^[0-9]+$ ]]; then
  echo "start index must be a non-negative integer" >&2
  exit 1
fi
if [[ ! "$num_tasks" =~ ^[1-9][0-9]*$ ]]; then
  echo "task count must be a positive integer" >&2
  exit 1
fi
if [[ -e "$bootstrap" ]]; then
  echo "refusing to overwrite bootstrap artifact: $bootstrap" >&2
  exit 1
fi

cd "$root"
uv run --no-sync scripts/build_q35_2b_environment_bootstrap_context_v1.py \
  --output "$bootstrap" \
  --axis "natural_n1a:$start_index" \
  --tasks-per-axis "$num_tasks" \
  --master-seed 20260819 \
  --leak-level action_scaffold

DUAL_SCAFFOLD_PROFILE=tight_answer_free_child_reporting_v1 \
DUAL_EXTERNAL_MODEL=q35-2b-recursive-coordinator-return \
DUAL_LEAK_COORDINATOR_EXACT_ACTION=0 \
DUAL_LEAK_COORDINATOR_RETURN_ACTION=0 \
DUAL_TYPED_COORDINATOR_RETURN=0 \
DUAL_ROOT_COORDINATOR_CONTRACT=1 \
DUAL_LEAF_REPORTER_CONTRACT=1 \
DUAL_LEAF_INLINE_EVIDENCE=1 \
DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD=0 \
DUAL_TYPED_CHILD_REPORT=1 \
QWEN38_QUALIFICATION_OUTPUT_ROOT="$output_root" \
QWEN38_QUALIFICATION_AXES=natural_n1a \
QWEN38_QUALIFICATION_NUM_TASKS="$num_tasks" \
QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
QWEN38_QUALIFICATION_MAX_CONCURRENT=4 \
QWEN38_QUALIFICATION_START_INDEX="$start_index" \
QUALIFICATION_REASONING_EFFORT=xhigh \
QUALIFICATION_SAMPLING_SEED=20260819 \
QUALIFICATION_MASTER_SEED=20260819 \
QUALIFICATION_SAMPLING_TEMPERATURE=1.0 \
QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH="$bootstrap" \
PROCEDURAL_INTERACTION_CURRICULUM=e0c4_recursive_coordinator_return \
scripts/run_q35_2b_dual_policy_mastery_v1.sh \
  "$coordinator_model" "$child_model" "$label" "$revision"
