#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
generator=$root/deps/verifiers/datasets/procedural_harness_master_v1/generate.py
template=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c160-child-live-context-compute-v5
corpus=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c160-child-balanced-live-compute-v6
candidate=/home/ubuntu/rlm/outputs/q35-2b-recursive-coordinator-return-v1/c160-child-balanced-live-compute-v6/weights/step_8
coordinator=/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1/grpo-auto-000158-coordinator-e0d3-uncapped-yield-exact-child-9415800/weights/step_1
train_start=9425000
eval_start=9425400
label=c158-c160-child-balanced-live-compute-v6-step8-9425400-n6
bootstrap=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c-return-9425400-n6-root-action-bootstrap.json

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -e "$corpus" ]]; then
  python scripts/build_q35_2b_balanced_live_child_compute_v1.py \
    --generator "$generator" --template-corpus "$template" \
    --start-index "$train_start" --examples-per-family 32 --output-dir "$corpus"
fi
jq -e '
  .status == "complete"
  and .objective == "balanced_answer_free_compute_in_exact_live_child_context"
  and .row_count == 224
  and .unique_task_count == .row_count
  and .task_index_range.selected_max < 9425400
  and .context_contract.roles == ["system", "user", "user", "assistant", "tool"]
  and .context_contract.leaf_reporter_contract == true
  and .context_contract.inline_evidence_target == true
  and .context_contract.answer_free_target == true
  and .context_contract.replay_rows == 0
  and ((.resource_family_counts | to_entries | map(.value) | unique) == [32])
  and (.operation_counts["count exact '\''green'\'' tokens"] > 0)
  and (.operation_counts["count exact '\''retry'\'' tokens"] > 0)
  and (.operation_counts["count exact '\''stable'\'' tokens"] > 0)
' "$corpus/MANIFEST.json" >/dev/null
expected_corpus_sha=$(jq -er '.dataset.sha256' "$corpus/MANIFEST.json")
actual_corpus_sha=$(sha256sum "$corpus/train.parquet" | awk '{print $1}')
if [[ "$actual_corpus_sha" != "$expected_corpus_sha" ]]; then
  echo "balanced live compute corpus checksum mismatch" >&2
  exit 1
fi
if [[ ! -f "$candidate/STABLE" || ! -f "$candidate/model.safetensors" ]]; then
  sft @ experiments/qwen35-2b-recursive-coordinator-return-v1/c160-child-balanced-live-compute-v6.toml
fi
test -f "$candidate/STABLE" -a -f "$candidate/model.safetensors"
python scripts/build_q35_2b_environment_bootstrap_context_v1.py \
  --output "$bootstrap" --axis natural_n1a:"$eval_start" --tasks-per-axis 6 \
  --leak-level action_scaffold

QWEN38_QUALIFICATION_CONFIG="$root/experiments/qwen35-2b-recursive-coordinator-return-v1/qualification-template.toml" \
QWEN38_QUALIFICATION_OUTPUT_ROOT=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1 \
QWEN38_QUALIFICATION_AXES=natural_n1a \
QWEN38_QUALIFICATION_NUM_TASKS=6 \
QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
QWEN38_QUALIFICATION_START_INDEX="$eval_start" \
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
