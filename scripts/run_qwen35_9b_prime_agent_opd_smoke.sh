#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${PRIME_AGENT_OPD_CONFIG:-$root/configs/debug/subagent-communication/329-qwen35-9b-prime-agent-mastery-opd.toml}
teacher=${PRIME_AGENT_OPD_TEACHER:?set PRIME_AGENT_OPD_TEACHER to a qualified stable 27B checkpoint}
output=${PRIME_AGENT_OPD_OUTPUT:-/ephemeral/subagent-rung/outputs/329-qwen35-9b-prime-agent-mastery-opd-smoke}
max_steps=${PRIME_AGENT_OPD_MAX_STEPS:-1}
teacher_log=${PRIME_AGENT_OPD_TEACHER_LOG:-/ephemeral/subagent-rung/logs/329-qwen35-27b-opd-teacher.log}

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -x .venv/bin/rl || ! -x .venv/bin/inference ]]; then
  echo "Prime-RL training executables are missing; run scripts/setup_prime_agent_mastery_host.sh" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "OPD config does not exist: $config" >&2
  exit 1
fi
if [[ ! -f "$teacher/STABLE" ]]; then
  echo "qualified teacher must be a local stable checkpoint: $teacher" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite OPD output: $output" >&2
  exit 1
fi
if (( max_steps < 1 )); then
  echo "PRIME_AGENT_OPD_MAX_STEPS must be positive" >&2
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
mkdir -p "$(dirname "$teacher_log")"

teacher_pid=
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$teacher_pid" ]]; then
    kill "$teacher_pid" 2>/dev/null || true
    wait "$teacher_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=6,7 inference \
  --router None \
  --server.port 8001 \
  --vllm.model "$teacher" \
  --vllm.tensor-parallel-size 2 \
  --vllm.max-model-len 65536 \
  --vllm.gpu-memory-utilization 0.80 \
  --vllm.max-num-seqs 4 \
  --vllm.enforce-eager true \
  --vllm.tool-call-parser qwen3_coder \
  --vllm.reasoning-parser qwen3 >"$teacher_log" 2>&1 &
teacher_pid=$!

for _ in $(seq 1 480); do
  if ! kill -0 "$teacher_pid" 2>/dev/null; then
    echo "teacher inference exited before becoming healthy; see $teacher_log" >&2
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8001/health >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -fsS http://127.0.0.1:8001/health >/dev/null; then
  echo "teacher inference did not become healthy within 480 seconds; see $teacher_log" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 rl @ "$config" \
  --max-steps "$max_steps" \
  --output-dir "$output" \
  --orchestrator.algo.teacher.name "$teacher" \
  --orchestrator.algo.teacher.base-url http://127.0.0.1:8001/v1
