#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator=/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1/grpo-auto-000158-coordinator-e0d3-uncapped-yield-exact-child-9415800/weights/step_1
child=/home/ubuntu/rlm/outputs/q35-2b-recursive-coordinator-return-v1/c160-child-compute-mix-v3/weights/step_10
tasks=${1:-24}
start=${2:-9424710}
label=${3:-c158-c160-step10-compute-report-curriculum-${start}-n${tasks}}
output_root=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1
bootstrap=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/${label}-root-action-bootstrap.json

if ! [[ "$tasks" =~ ^[1-9][0-9]*$ && "$start" =~ ^[0-9]+$ ]]; then
  echo "tasks must be positive and start must be nonnegative" >&2
  exit 1
fi

cd "$root"
export PATH="$root/.venv/bin:$PATH"
python scripts/build_q35_2b_environment_bootstrap_context_v1.py \
  --output "$bootstrap" --axis natural_n1a:"$start" --tasks-per-axis "$tasks" \
  --leak-level action_scaffold

QWEN38_QUALIFICATION_CONFIG="$root/experiments/qwen35-2b-recursive-coordinator-return-v1/qualification-template.toml" \
QWEN38_QUALIFICATION_OUTPUT_ROOT="$output_root" \
QWEN38_QUALIFICATION_AXES=natural_n1a \
QWEN38_QUALIFICATION_NUM_TASKS="$tasks" \
QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
QWEN38_QUALIFICATION_START_INDEX="$start" \
QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH="$bootstrap" \
PROCEDURAL_INTERACTION_CURRICULUM=e0c4_recursive_coordinator_return \
PROCEDURAL_NATURAL_YIELD_SCAFFOLD=1 \
DUAL_EXTERNAL_MODEL=q35-2b-recursive-coordinator-return \
DUAL_ROOT_COORDINATOR_CONTRACT=1 \
DUAL_LEAF_REPORTER_CONTRACT=1 \
DUAL_LEAF_INLINE_EVIDENCE=1 \
DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD=1 \
UV_BIN=/home/ubuntu/.local/bin/uv \
scripts/run_q35_2b_dual_policy_mastery_v1.sh \
  "$coordinator" "$child" "$label" candidate-local

run_dir=$output_root/$label
traces=$run_dir/natural_n1a/traces.jsonl
jq -s --argjson requested "$tasks" '{
  requested: $requested,
  episodes: length,
  errors: ([.[] | select((.errors | length) > 0)] | length),
  child_completed: ([.[] | select(.traces[0].metrics.child_action_completed == 1)] | length),
  hard_successes: ([.[] | select(
    .traces[0].rewards.harness_score.score == 1
    and .traces[0].metrics.child_action_completed == 1
    and .traces[0].stop_condition == "user_closed"
  )] | length),
  purpose: "scaffolded curriculum harvest; never admission evidence"
}' "$traces" | tee "$run_dir/CURRICULUM_SUMMARY.json"
