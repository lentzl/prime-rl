#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
harvest_label=c158-c160-step4-natural-compute-harvest-9423900-n24
harvest_root=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/$harvest_label
harvest_traces=$harvest_root/natural_n1a/traces.jsonl
replay_anchor=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c160-child-return40-sft-v1
corpus=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c160-child-natural-compute-replay-v2
training_config=$root/experiments/qwen35-2b-recursive-coordinator-return-v1/c160-child-natural-compute-replay-v2.toml
candidate=/home/ubuntu/rlm/outputs/q35-2b-recursive-coordinator-return-v1/c160-child-natural-compute-replay-v2/weights/step_8
coordinator=/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1/grpo-auto-000158-coordinator-e0d3-uncapped-yield-exact-child-9415800/weights/step_1
screen_start=9424200
screen_label=c158-c160-child-natural-compute-replay-v2-step8-9424200-n6
screen_bootstrap=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c-return-9424200-n6-root-action-bootstrap.json

cd "$root"
export PATH="$root/.venv/bin:$PATH"

if [[ ! -s "$harvest_traces" || "$(wc -l <"$harvest_traces")" != 24 ]]; then
  echo "natural child harvest is incomplete: $harvest_traces" >&2
  exit 1
fi
if jq -e 'select((.errors | length) > 0)' "$harvest_traces" >/dev/null; then
  echo "natural child harvest contains evaluator errors" >&2
  exit 1
fi
accepted=$(
  jq -s '[.[] | select(
    .traces[0].metrics.child_action_completed == 1
    and .traces[0].stop_condition == "user_closed"
  )] | length' "$harvest_traces"
)
if ((accepted < 1)); then
  echo "natural child harvest contains no complete role-scoped deliveries" >&2
  exit 1
fi
return_repeats=$(((40 + accepted - 1) / accepted))

python scripts/build_q35_2b_recursive_return_trace_sft_v1.py \
  --forced-return-traces "$harvest_traces" \
  --child-only \
  --natural-child-actions \
  --replay-anchor-corpus "$replay_anchor" \
  --output-dir "$corpus" \
  --return-repeats "$return_repeats" \
  --minimum-return-traces 1

sft @ "$training_config"
if [[ ! -f "$candidate/STABLE" || ! -f "$candidate/model.safetensors" ]]; then
  echo "natural compute replay candidate is incomplete: $candidate" >&2
  exit 1
fi

python scripts/build_q35_2b_environment_bootstrap_context_v1.py \
  --output "$screen_bootstrap" \
  --axis natural_n1a:"$screen_start" \
  --tasks-per-axis 6 \
  --leak-level action_scaffold

QWEN38_QUALIFICATION_CONFIG="$root/experiments/qwen35-2b-recursive-coordinator-return-v1/qualification-template.toml" \
QWEN38_QUALIFICATION_OUTPUT_ROOT=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1 \
QWEN38_QUALIFICATION_AXES=natural_n1a \
QWEN38_QUALIFICATION_NUM_TASKS=6 \
QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
QWEN38_QUALIFICATION_START_INDEX="$screen_start" \
QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH="$screen_bootstrap" \
PROCEDURAL_INTERACTION_CURRICULUM=e0c4_recursive_coordinator_return \
DUAL_EXTERNAL_MODEL=q35-2b-recursive-coordinator-return \
DUAL_ROOT_COORDINATOR_CONTRACT=1 \
DUAL_LEAF_REPORTER_CONTRACT=1 \
DUAL_LEAF_INLINE_EVIDENCE=1 \
DUAL_LEAK_COORDINATOR_EXACT_ACTION=1 \
UV_BIN=/home/ubuntu/.local/bin/uv \
scripts/run_q35_2b_dual_policy_mastery_v1.sh \
  "$coordinator" "$candidate" "$screen_label" candidate-local

screen_traces=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/$screen_label/natural_n1a/traces.jsonl
jq -s '{
  episodes: length,
  errors: ([.[] | select((.errors | length) > 0)] | length),
  qualifying: ([.[] | select(
    .traces[0].rewards.harness_score.score == 1
    and .traces[0].metrics.child_action_completed == 1
    and .traces[0].stop_condition == "user_closed"
  )] | length),
  admission_floor: 4
}' "$screen_traces"
