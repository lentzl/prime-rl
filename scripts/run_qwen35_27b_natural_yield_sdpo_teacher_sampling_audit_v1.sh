#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=${NATURAL_YIELD_SDPO_AUDIT_RUN_DIR:-/ephemeral/outputs/qwen35-27b-natural-yield-sdpo-v1/teacher-distribution-audit-v1-rerun}
model=${NATURAL_YIELD_SDPO_MODEL:-/ephemeral/outputs/qwen35-27b-procedural-harness-action-ramp-v1/atomic-send-grpo-r7/weights/step_1}
cuda_devices=${TEACHER_SAMPLING_CUDA_VISIBLE_DEVICES:-0,1}
tensor_parallel_size=${TEACHER_SAMPLING_TENSOR_PARALLEL_SIZE:-2}
router_port=${TEACHER_SAMPLING_ROUTER_PORT:-8000}
backend_port=${TEACHER_SAMPLING_BACKEND_PORT:-8100}
max_model_len=${TEACHER_SAMPLING_MAX_MODEL_LEN:-12288}
samples_per_arm=${TEACHER_SAMPLING_SAMPLES_PER_ARM:-8}
expected_state_count=${TEACHER_SAMPLING_EXPECTED_STATE_COUNT:-8}
max_tokens=${TEACHER_SAMPLING_MAX_TOKENS:-1536}
temperature=${TEACHER_SAMPLING_TEMPERATURE:-1.0}
sampling_timeout=${TEACHER_SAMPLING_TIMEOUT:-1200}
output=$run_dir/TEACHER_SAMPLING.json
decision=$run_dir/TEACHER_ADMISSION.json
service_dir=$run_dir/teacher_sampling_service
serve_config=$service_dir/inference.toml
serve_log=$service_dir/inference.log
inference_bin=${INFERENCE_BIN:-$root/.venv/bin/inference}
router_bin=${VLLM_ROUTER_BIN:-$root/.venv/bin/vllm-router}
curl_bin=${CURL_BIN:-curl}
nvidia_smi_bin=${NVIDIA_SMI_BIN:-nvidia-smi}

cd "$root"
export PATH="$(dirname "$router_bin"):$root/.venv/bin:$PATH"

if [[ ! -x "$inference_bin" || ! -x "$router_bin" ]]; then
  echo "Prime-RL inference or vllm-router executable is missing" >&2
  exit 1
fi
if [[ ! -f "$run_dir/AUDIT.json" ]]; then
  echo "distribution audit is incomplete: $run_dir/AUDIT.json" >&2
  exit 1
fi
if ! compgen -G "$run_dir/token_exports/step_1/rank_*.jsonl" >/dev/null; then
  echo "distribution audit has no token exports" >&2
  exit 1
fi
if [[ ! -f "$model/STABLE" ]]; then
  echo "canonical model has no STABLE marker: $model" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite teacher sampling report: $output" >&2
  exit 1
fi
if [[ -e "$decision" ]]; then
  echo "refusing to overwrite teacher admission decision: $decision" >&2
  exit 1
fi

IFS=, read -ra devices <<<"$cuda_devices"
if [[ ${#devices[@]} -ne $tensor_parallel_size ]]; then
  echo "CUDA device count must equal tensor parallel size" >&2
  exit 1
fi
for device in "${devices[@]}"; do
  if [[ -n "$("$nvidia_smi_bin" --id="$device" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "refusing to launch while GPU $device is occupied" >&2
    exit 1
  fi
done

mkdir -p "$service_dir"
cat >"$serve_config" <<EOF
backend_port = $backend_port

[server]
port = $router_port
liveness_timeout_seconds = 30.0

[vllm]
model = "$model"
dtype = "auto"
max_model_len = $max_model_len
enforce_eager = true
trust_remote_code = false
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"
tensor_parallel_size = $tensor_parallel_size
data_parallel_size = 1
data_parallel_rpc_port = 13345
gpu_memory_utilization = 0.80
max_num_seqs = $samples_per_arm

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

CUDA_VISIBLE_DEVICES=$cuda_devices "$inference_bin" @ "$serve_config" >"$serve_log" 2>&1 &
inference_pid=$!
for _ in $(seq 1 480); do
  if ! kill -0 "$inference_pid" 2>/dev/null; then
    echo "inference exited before becoming healthy; see $serve_log" >&2
    exit 1
  fi
  if "$curl_bin" -fsS "http://127.0.0.1:$backend_port/health" >/dev/null; then
    break
  fi
  sleep 1
done
if ! "$curl_bin" -fsS "http://127.0.0.1:$backend_port/health" >/dev/null; then
  echo "inference did not become healthy within 480 seconds; see $serve_log" >&2
  exit 1
fi

"$root/.venv/bin/python" -m scripts.sample_natural_yield_sdpo_teacher_replays_v1 \
  "$run_dir" \
  --base-url "http://127.0.0.1:$backend_port/v1" \
  --model "$model" \
  --samples-per-arm "$samples_per_arm" \
  --expected-state-count "$expected_state_count" \
  --max-tokens "$max_tokens" \
  --temperature "$temperature" \
  --timeout "$sampling_timeout" \
  --output "$output"

"$root/.venv/bin/python" -m scripts.decide_natural_yield_sdpo_teacher_admission_v1 \
  "$run_dir/AUDIT.json" \
  "$output" \
  --output "$decision"

echo "teacher sampling audit written to $output"
echo "teacher admission decision written to $decision"
