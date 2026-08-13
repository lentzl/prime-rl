#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:?usage: run_qwen35_27b_mastery_fast_screen_model_v1.sh MODEL LABEL [REVISION]}
label=${2:?usage: run_qwen35_27b_mastery_fast_screen_model_v1.sh MODEL LABEL [REVISION]}
revision=${3:-}
output_root=${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/subagent-rung/evals/320-326-qwen35-27b-mastery-fast-screen-v1}
run_output=$output_root/$label
serve_config=$run_output/inference.toml
serve_log=$run_output/inference.log
cuda_devices=${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}
tensor_parallel_size=${EVAL_TENSOR_PARALLEL_SIZE:-4}

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -x .venv/bin/inference || ! -x .venv/bin/eval ]]; then
  echo "Prime-RL inference/eval executables are missing" >&2
  exit 1
fi
if [[ -e "$run_output" ]]; then
  echo "refusing to overwrite fast-screen output: $run_output" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi
if [[ "$model" = /* ]] && [[ ! -f "$model/STABLE" ]]; then
  echo "local model has no STABLE marker: $model" >&2
  exit 1
fi

set -a
source .env
set +a
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
mkdir -p "$run_output"

revision_line=
if [[ -n "$revision" ]]; then
  revision_line="revision = \"$revision\""
fi
cat >"$serve_config" <<EOF
backend_port = 8100

[server]
port = 8000
liveness_timeout_seconds = 30.0

[vllm]
model = "$model"
dtype = "auto"
max_model_len = 65536
enforce_eager = true
trust_remote_code = false
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"
tensor_parallel_size = $tensor_parallel_size
data_parallel_size = 1
gpu_memory_utilization = 0.80
max_num_seqs = 4
$revision_line

[log]
level = "info"
vf_level = "info"
json_logging = false
log_data = false
interval = 10.0
EOF

inference_pid=
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$inference_pid" ]]; then
    kill "$inference_pid" 2>/dev/null || true
    wait "$inference_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=$cuda_devices inference @ "$serve_config" >"$serve_log" 2>&1 &
inference_pid=$!
for _ in $(seq 1 480); do
  if ! kill -0 "$inference_pid" 2>/dev/null; then
    echo "inference exited before becoming healthy; see $serve_log" >&2
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8100/health >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -fsS http://127.0.0.1:8100/health >/dev/null; then
  echo "inference did not become healthy within 480 seconds; see $serve_log" >&2
  exit 1
fi

PRIME_MASTERY_OUTPUT_ROOT=$output_root \
  scripts/run_qwen35_27b_mastery_fast_screen_v1.sh "$model" "$label"
echo "fast mastery screen completed: $run_output"
