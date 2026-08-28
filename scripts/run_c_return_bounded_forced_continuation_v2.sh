#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1/grpo-auto-000158-coordinator-e0d3-uncapped-yield-exact-child-9415800/weights/step_1
revision=18f760904ba4ef6b54b785dea96307e100d8ff71719750d8a76fabd9a55a789e
output_root=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1
artifact_root=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1
root_anchors=/home/ubuntu/rlm/results/q35-2b-self-bootstrap-dual-dense-grpo-v1/grpo-auto-000158-coordinator-e0d3-uncapped-yield-exact-child-9415800-admission-9415850-n6/natural_n1a/traces.jsonl
aggregate=$artifact_root/c158-forced-return-bounded24-v2.jsonl
corpus=$artifact_root/c158-forced-return28-root8-sft-v2
training_config=$root/experiments/qwen35-2b-recursive-coordinator-return-v1/c158-c-return-sft-forced28-root8-v2.toml
candidate=/home/ubuntu/rlm/outputs/q35-2b-recursive-coordinator-return-v1/c158-c-return-sft-forced28-root8-v2/weights/step_6
starts=(9422400 9422500 9422600 9422700)

cd "$root"
export PATH="$root/.venv/bin:$PATH"

if [[ -e "$aggregate" || -e "$corpus" || -e "${candidate%/weights/step_6}" ]]; then
  echo "refusing to overwrite a bounded continuation artifact or training output" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to start bounded continuation while a GPU process is active" >&2
  exit 1
fi
mkdir -p "$artifact_root"

trace_files=()
for start in "${starts[@]}"; do
  label="c158-c-return-forced-bounded-${start}-n6"
  bootstrap="$artifact_root/c-return-${start}-n6-strategy_hint-bootstrap.json"
  python scripts/build_q35_2b_environment_bootstrap_context_v1.py \
    --output "$bootstrap" \
    --axis "natural_n1a:${start}" \
    --tasks-per-axis 6 \
    --leak-level strategy_hint
  C_RETURN_QUALIFICATION_CONFIG="$root/experiments/qwen35-2b-recursive-coordinator-return-v1/qualification-action-scaffold.toml" \
  C_RETURN_START_INDEX="$start" \
  C_RETURN_NUM_TASKS=6 \
  DUAL_LEAK_COORDINATOR_RETURN_ACTION=1 \
    scripts/run_q35_2b_recursive_coordinator_return_v1.sh \
      "$model" "$label" "$revision" "$bootstrap"
  traces="$output_root/$label/natural_n1a/traces.jsonl"
  episodes=$(wc -l <"$traces")
  errors=$(jq -s '[.[] | .traces[0] | select(.stop_condition == "error")] | length' "$traces")
  qualifying=$(
    jq -s '[.[] | .traces[0] | select(
      .rewards.harness_score.score == 1
      and .metrics.child_action_completed == 1
      and .stop_condition == "user_closed"
    )] | length' "$traces"
  )
  if [[ "$episodes" != 6 || "$errors" != 0 || "$qualifying" -lt 4 ]]; then
    echo "bounded forced bank rejected: label=$label episodes=$episodes errors=$errors qualifying=$qualifying" >&2
    exit 1
  fi
  trace_files+=("$traces")
done

for traces in "${trace_files[@]}"; do
  jq -c '. | select(
    .traces[0].rewards.harness_score.score == 1
    and .traces[0].metrics.child_action_completed == 1
    and .traces[0].stop_condition == "user_closed"
  )' "$traces" >>"$aggregate"
done
qualifying=$(wc -l <"$aggregate")
if ((qualifying < 18)); then
  echo "bounded forced aggregate has too few hard successes: $qualifying" >&2
  exit 1
fi

python scripts/build_q35_2b_recursive_return_trace_sft_v1.py \
  --forced-return-traces "$aggregate" \
  --root-anchor-traces "$root_anchors" \
  --output-dir "$corpus" \
  --return-repeats 1 \
  --root-anchor-repeats 2 \
  --minimum-return-traces 18

sft @ "$training_config"
if [[ ! -f "$candidate/STABLE" || ! -f "$candidate/model.safetensors" ]]; then
  echo "bounded coordinator candidate is incomplete: $candidate" >&2
  exit 1
fi

screen_start=9422800
screen_label=c158-c-return-sft-bounded-v2-step6-natural-screen-9422800-n6
screen_bootstrap=$artifact_root/c-return-9422800-n6-strategy_hint-bootstrap.json
python scripts/build_q35_2b_environment_bootstrap_context_v1.py \
  --output "$screen_bootstrap" \
  --axis natural_n1a:"$screen_start" \
  --tasks-per-axis 6 \
  --leak-level strategy_hint
C_RETURN_START_INDEX="$screen_start" \
C_RETURN_NUM_TASKS=6 \
  scripts/run_q35_2b_recursive_coordinator_return_v1.sh \
    "$candidate" "$screen_label" candidate-local "$screen_bootstrap"

screen_traces=$output_root/$screen_label/natural_n1a/traces.jsonl
jq -s '{
  episodes: length,
  qualifying: ([.[] | .traces[0] | select(
    .rewards.harness_score.score == 1
    and .metrics.child_action_completed == 1
    and .stop_condition == "user_closed"
  )] | length),
  admission_floor: 4
}' "$screen_traces"
