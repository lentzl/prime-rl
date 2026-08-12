#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
OUTPUT_ROOT=/ephemeral/subagent-rung/evals/267-qwen35-27b-native-sibling-repeat-r1
OWNERSHIP_CONFIG=configs/debug/subagent-communication/267-qwen35-27b-native-sibling-ownership-repeat.toml
NATURAL_CONFIG=configs/debug/subagent-communication/239-qwen35-27b-turn-opsd-repeat.toml
ADAPTER_NAME=qwen35-27b-native-sibling-step1-repeat
ADAPTER_PATH=/ephemeral/subagent-rung/outputs/265-qwen35-27b-native-sibling-sdpo-dose-r1/weights/step_1

cd "$ROOT"
test -d "$ADAPTER_PATH"
rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

systemctl --user restart qwen35-27b-teacher-inference.service
for _ in $(seq 1 150); do
  if curl --fail --silent http://127.0.0.1:8100/health >/dev/null \
    && curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:8100/health >/dev/null
curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null

/home/ubuntu/prime-rl/.venv/bin/eval @ "$OWNERSHIP_CONFIG" --model Qwen/Qwen3.5-27B \
  --output-dir "$OUTPUT_ROOT/base-child"
/home/ubuntu/prime-rl/.venv/bin/eval @ "$OWNERSHIP_CONFIG" --model Qwen/Qwen3.5-27B \
  --env.taskset.ownership coordinator --output-dir "$OUTPUT_ROOT/base-direct"
/home/ubuntu/prime-rl/.venv/bin/eval @ "$NATURAL_CONFIG" --model Qwen/Qwen3.5-27B \
  --output-dir "$OUTPUT_ROOT/base-natural"

curl --fail --silent --show-error \
  --request POST http://127.0.0.1:8100/v1/load_lora_adapter \
  --header 'Content-Type: application/json' \
  --data "{\"lora_name\":\"$ADAPTER_NAME\",\"lora_path\":\"$ADAPTER_PATH\"}"
printf '\n'

/home/ubuntu/prime-rl/.venv/bin/eval @ "$OWNERSHIP_CONFIG" --model "$ADAPTER_NAME" \
  --output-dir "$OUTPUT_ROOT/candidate-child"
/home/ubuntu/prime-rl/.venv/bin/eval @ "$OWNERSHIP_CONFIG" --model "$ADAPTER_NAME" \
  --env.taskset.ownership coordinator --output-dir "$OUTPUT_ROOT/candidate-direct"
/home/ubuntu/prime-rl/.venv/bin/eval @ "$NATURAL_CONFIG" --model "$ADAPTER_NAME" \
  --output-dir "$OUTPUT_ROOT/candidate-natural"

/home/ubuntu/prime-rl/.venv/bin/python scripts/summarize_ownership_candidate_selection.py \
  --child-base "$OUTPUT_ROOT/base-child/traces.jsonl" \
  --child-candidate "$OUTPUT_ROOT/candidate-child/traces.jsonl" \
  --direct-base "$OUTPUT_ROOT/base-direct/traces.jsonl" \
  --direct-candidate "$OUTPUT_ROOT/candidate-direct/traces.jsonl" \
  --output "$OUTPUT_ROOT/ownership-selection.json"
/home/ubuntu/prime-rl/.venv/bin/python scripts/summarize_natural_control_selection.py \
  --base "$OUTPUT_ROOT/base-natural/traces.jsonl" \
  --candidate "$OUTPUT_ROOT/candidate-natural/traces.jsonl" \
  --output "$OUTPUT_ROOT/natural-selection.json"
