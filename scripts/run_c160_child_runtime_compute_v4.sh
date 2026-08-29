#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
harvest_a=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/c158-c160-step10-compute-report-curriculum-9424710-n24/natural_n1a/traces.jsonl
harvest_b=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/c158-c160-step10-compute-report-curriculum-resume-9424723-n11/natural_n1a/traces.jsonl
replay=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c160-child-natural-compute-replay-v2
corpus=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c160-child-runtime-compute-v4
candidate=/home/ubuntu/rlm/outputs/q35-2b-recursive-coordinator-return-v1/c160-child-runtime-compute-v4/weights/step_8
coordinator=/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1/grpo-auto-000158-coordinator-e0d3-uncapped-yield-exact-child-9415800/weights/step_1
start=9424800
label=c158-c160-child-runtime-compute-v4-step8-9424800-n6
bootstrap=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c-return-9424800-n6-root-action-bootstrap.json

cd "$root"
export PATH="$root/.venv/bin:$PATH"
python scripts/build_q35_2b_recursive_return_trace_sft_v1.py \
  --forced-return-traces "$harvest_a" \
  --forced-return-traces "$harvest_b" \
  --child-only --scaffolded-compute-actions \
  --replay-anchor-corpus "$replay" \
  --return-repeats 3 --replay-anchor-repeats 1 \
  --minimum-return-traces 16 --output-dir "$corpus"
sft @ experiments/qwen35-2b-recursive-coordinator-return-v1/c160-child-runtime-compute-v4.toml
test -f "$candidate/STABLE" -a -f "$candidate/model.safetensors"
python scripts/build_q35_2b_environment_bootstrap_context_v1.py \
  --output "$bootstrap" --axis natural_n1a:"$start" --tasks-per-axis 6 \
  --leak-level action_scaffold

QWEN38_QUALIFICATION_CONFIG="$root/experiments/qwen35-2b-recursive-coordinator-return-v1/qualification-template.toml" \
QWEN38_QUALIFICATION_OUTPUT_ROOT=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1 \
QWEN38_QUALIFICATION_AXES=natural_n1a \
QWEN38_QUALIFICATION_NUM_TASKS=6 \
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
UV_BIN=/home/ubuntu/.local/bin/uv \
scripts/run_q35_2b_dual_policy_mastery_v1.sh \
  "$coordinator" "$candidate" "$label" candidate-local

traces=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/$label/natural_n1a/traces.jsonl
jq -s '{
  episodes: length,
  errors: ([.[] | select((.errors | length) > 0)] | length),
  qualifying: ([.[] | select(
    .traces[0].rewards.harness_score.score == 1
    and .traces[0].metrics.child_action_completed == 1
    and .traces[0].stop_condition == "user_closed"
  )] | length),
  admission_floor: 4,
  admitted: (([.[] | select(
    .traces[0].rewards.harness_score.score == 1
    and .traces[0].metrics.child_action_completed == 1
    and .traces[0].stop_condition == "user_closed"
  )] | length) >= 4)
}' "$traces"
