#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
CONFIG=configs/debug/subagent-communication/269-qwen35-27b-native-sibling-continuation.toml
TRAIN_ROOT=/ephemeral/subagent-rung/outputs/269-qwen35-27b-native-sibling-continuation-r1
EVAL_ROOT=/ephemeral/subagent-rung/evals/269-qwen35-27b-native-sibling-continuation-r1
BASE_ROOT=/ephemeral/subagent-rung/evals/266-qwen35-27b-native-sibling-selection-r1
OWNERSHIP_CONFIG=configs/debug/subagent-communication/266-qwen35-27b-native-sibling-ownership-selection.toml
NATURAL_CONFIG=configs/debug/subagent-communication/258-qwen35-27b-action-local-selection.toml
INFERENCE_SERVICE=qwen35-27b-teacher-inference.service

cd "$ROOT"
export PATH="$ROOT/.venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
export HF_TOKEN="$HF_KEY"
export CUDA_VISIBLE_DEVICES=2,3
trap 'systemctl --user stop "$INFERENCE_SERVICE"' EXIT
rm -rf "$EVAL_ROOT"
mkdir -p "$EVAL_ROOT/frozen-base"
cp "$BASE_ROOT/base-child/traces.jsonl" "$EVAL_ROOT/frozen-base/base-child.jsonl"
cp "$BASE_ROOT/base-direct/traces.jsonl" "$EVAL_ROOT/frozen-base/base-direct.jsonl"
cp "$BASE_ROOT/base-natural/traces.jsonl" "$EVAL_ROOT/frozen-base/base-natural.jsonl"

wait_for_inference() {
  for _ in $(seq 1 150); do
    if curl --fail --silent http://127.0.0.1:8100/health >/dev/null \
      && curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then
      return
    fi
    sleep 2
  done
  return 1
}

evaluate_step() {
  local step=$1
  local output="$EVAL_ROOT/step-$step"
  local adapter_name="qwen35-27b-native-continuation-step-$step"
  mkdir -p "$output"

  systemctl --user restart "$INFERENCE_SERVICE"
  wait_for_inference
  curl --fail --silent --show-error \
    --request POST http://127.0.0.1:8100/v1/load_lora_adapter \
    --header 'Content-Type: application/json' \
    --data "{\"lora_name\":\"$adapter_name\",\"lora_path\":\"$TRAIN_ROOT/weights/step_$step\"}"
  printf '\n'

  .venv/bin/eval @ "$OWNERSHIP_CONFIG" --model "$adapter_name" \
    --output-dir "$output/candidate-child"
  .venv/bin/eval @ "$OWNERSHIP_CONFIG" --model "$adapter_name" \
    --env.taskset.ownership coordinator --output-dir "$output/candidate-direct"
  .venv/bin/eval @ "$NATURAL_CONFIG" --model "$adapter_name" \
    --output-dir "$output/candidate-natural"

  .venv/bin/python scripts/summarize_ownership_candidate_selection.py \
    --child-base "$BASE_ROOT/base-child/traces.jsonl" \
    --child-candidate "$output/candidate-child/traces.jsonl" \
    --direct-base "$BASE_ROOT/base-direct/traces.jsonl" \
    --direct-candidate "$output/candidate-direct/traces.jsonl" \
    --output "$output/ownership-selection.json"
  .venv/bin/python scripts/summarize_natural_control_selection.py \
    --base "$BASE_ROOT/base-natural/traces.jsonl" \
    --candidate "$output/candidate-natural/traces.jsonl" \
    --output "$output/natural-selection.json"
  .venv/bin/python scripts/summarize_native_sibling_continuation.py \
    --root "$EVAL_ROOT" --output "$EVAL_ROOT/trajectory-summary.json"
}

for step in 1 2 3 4; do
  systemctl --user restart "$INFERENCE_SERVICE"
  wait_for_inference
  if [[ "$step" == 1 ]]; then
    .venv/bin/rl @ "$CONFIG" --max-steps 1
  else
    previous=$((step - 1))
    .venv/bin/rl @ "$CONFIG" --max-steps "$step" --ckpt.resume-step "$previous" --no-clean-output-dir
  fi
  test -e "$TRAIN_ROOT/weights/step_$step/STABLE"
  evaluate_step "$step"

  classification=$(jq -r .classification "$EVAL_ROOT/trajectory-summary.json")
  if [[ "$classification" == PROMOTE_CANDIDATE || "$classification" == BRANCH_REJECTED_HARD_INVARIANT ]]; then
    break
  fi
done
