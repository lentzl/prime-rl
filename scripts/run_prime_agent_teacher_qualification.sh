#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_output=${TEACHER_SOURCE_OUTPUT:-/ephemeral/subagent-rung/outputs/283-qwen35-27b-prime-agent-teacher-bootstrap-online-r64}
source_config=${TEACHER_SOURCE_CONFIG:-$root/configs/debug/subagent-communication/283-qwen35-27b-prime-agent-teacher-bootstrap-online.toml}
output=${TEACHER_QUALIFICATION_OUTPUT:-/ephemeral/subagent-rung/evals/315-qwen35-27b-prime-agent-teacher-checkpoint-qualification}

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -x .venv/bin/inference || ! -x .venv/bin/evaluator || ! -x .venv/bin/env-server ]]; then
  echo "Prime-RL inference/evaluation executables are missing" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite qualification output: $output" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

set -a
source .env
set +a
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN or HF_KEY is required" >&2
  exit 1
fi

.venv/bin/python scripts/prepare_prime_agent_teacher_qualification.py \
  "$source_config" "$source_output" "$output"
mkdir -p "$output/logs/envs/eval"

pids=()
names=()
cleanup() {
  trap - EXIT INT TERM
  for ((index=${#pids[@]}-1; index>=0; index--)); do
    kill "${pids[$index]}" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0,1,2,3 inference @ "$output/configs/inference.toml" \
  >"$output/logs/inference.log" 2>&1 &
pids+=("$!")
names+=("inference")

for config in "$output"/configs/envs/eval/*.toml; do
  name=$(basename "$config" .toml)
  env-server @ "$config" >"$output/logs/envs/eval/$name.log" 2>&1 &
  pids+=("$!")
  names+=("env/$name")
done

evaluator @ "$output/configs/evaluator.toml" >"$output/logs/evaluator.log" 2>&1 &
evaluator_pid=$!
pids+=("$evaluator_pid")
names+=("evaluator")

while kill -0 "$evaluator_pid" 2>/dev/null; do
  for ((index=0; index<${#pids[@]}-1; index++)); do
    if ! kill -0 "${pids[$index]}" 2>/dev/null; then
      echo "${names[$index]} exited before qualification completed" >&2
      exit 1
    fi
  done
  sleep 5
done

if ! wait "$evaluator_pid"; then
  echo "checkpoint qualification failed" >&2
  exit 1
fi

trace_paths=("$output"/rollouts/step_*/eval/all/traces.jsonl)
if [[ ! -e "${trace_paths[0]}" ]]; then
  echo "checkpoint qualification produced no trace files" >&2
  exit 1
fi
.venv/bin/python scripts/summarize_prime_agent_mastery.py \
  --json --by-policy-version "${trace_paths[@]}" \
  >"$output/qualification-summary.json"
echo "checkpoint qualification completed: $output"
