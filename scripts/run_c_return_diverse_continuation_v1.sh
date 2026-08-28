#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
collection_label=c158-c-return-forced-collection-9421300-n28
collection_root=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/$collection_label
collection_traces=$collection_root/natural_n1a/traces.jsonl
corpus=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c158-forced-return28-root8-sft-v2
root_anchors=/home/ubuntu/rlm/results/q35-2b-self-bootstrap-dual-dense-grpo-v1/grpo-auto-000158-coordinator-e0d3-uncapped-yield-exact-child-9415800-admission-9415850-n6/natural_n1a/traces.jsonl
training_config=$root/experiments/qwen35-2b-recursive-coordinator-return-v1/c158-c-return-sft-forced28-root8-v2.toml
candidate=/home/ubuntu/rlm/outputs/q35-2b-recursive-coordinator-return-v1/c158-c-return-sft-forced28-root8-v2/weights/step_6
screen_start=9421700
screen_label=c158-c-return-sft-forced28-root8-v2-step6-action-screen-9421700-n6
screen_bootstrap=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c-return-9421700-n6-strategy_hint-bootstrap.json

cd "$root"
export PATH="$root/.venv/bin:$PATH"

while pgrep -f "eval @ .*${collection_label}" >/dev/null; do
  sleep 30
done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
  sleep 5
done
if [[ ! -s "$collection_traces" ]]; then
  echo "collection trace artifact is missing: $collection_traces" >&2
  exit 1
fi
episodes=$(wc -l <"$collection_traces")
qualifying=$(
  jq -s '[.[] | select(
    .traces[0].rewards.harness_score.score == 1
    and .traces[0].metrics.child_action_completed == 1
    and .traces[0].stop_condition == "user_closed"
  )] | length' "$collection_traces"
)
if [[ "$episodes" != 28 || "$qualifying" != 28 ]]; then
  echo "forced collection incomplete: episodes=$episodes qualifying=$qualifying" >&2
  exit 1
fi

python scripts/build_q35_2b_recursive_return_trace_sft_v1.py \
  --forced-return-traces "$collection_traces" \
  --root-anchor-traces "$root_anchors" \
  --output-dir "$corpus" \
  --return-repeats 1 \
  --root-anchor-repeats 2

sft @ "$training_config"
if [[ ! -f "$candidate/STABLE" || ! -f "$candidate/model.safetensors" ]]; then
  echo "diverse coordinator candidate is incomplete: $candidate" >&2
  exit 1
fi

python scripts/build_q35_2b_environment_bootstrap_context_v1.py \
  --output "$screen_bootstrap" \
  --axis natural_n1a:"$screen_start" \
  --tasks-per-axis 6 \
  --leak-level strategy_hint

C_RETURN_QUALIFICATION_CONFIG="$root/experiments/qwen35-2b-recursive-coordinator-return-v1/qualification-action-scaffold.toml" \
C_RETURN_START_INDEX="$screen_start" \
scripts/run_q35_2b_recursive_coordinator_return_v1.sh \
  "$candidate" \
  "$screen_label" \
  candidate-local \
  "$screen_bootstrap"

screen_traces=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/$screen_label/natural_n1a/traces.jsonl
jq -s '{
  episodes: length,
  qualifying: ([.[] | select(
    .traces[0].rewards.harness_score.score == 1
    and .traces[0].metrics.child_action_completed == 1
  )] | length),
  admission_floor: 4
}' "$screen_traces"
