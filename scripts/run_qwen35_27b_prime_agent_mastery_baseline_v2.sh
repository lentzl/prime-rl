#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-untouched-base}
revision=${3:-fc05daec18b0a78c049392ed2e771dde82bdf654}
output_root=${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-prime-agent-mastery-v2}
run_output=$output_root/$label
serve_log=$run_output/inference.log
cuda_devices=${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}
tensor_parallel_size=${EVAL_TENSOR_PARALLEL_SIZE:-4}
backend_port=${EVAL_BACKEND_PORT:-8100}
router_port=${EVAL_ROUTER_PORT:-8000}
data_parallel_rpc_port=${EVAL_DATA_PARALLEL_RPC_PORT:-13345}
eval_driver=scripts/run_qwen35_27b_prime_agent_mastery_battery_v2.sh
inference_bin=${INFERENCE_BIN:-$root/.venv/bin/inference}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
nvidia_smi_bin=${NVIDIA_SMI_BIN:-nvidia-smi}
dry_run=${BASELINE_DRY_RUN:-false}

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -x "$inference_bin" || ! -x "$eval_bin" ]]; then
  echo "Prime-RL inference/eval executables are missing" >&2
  exit 1
fi
if [[ ! -x "$eval_driver" ]]; then
  echo "mastery eval driver is not executable: $eval_driver" >&2
  exit 1
fi
if [[ -z "$revision" ]]; then
  echo "a pinned model revision is required" >&2
  exit 1
fi
if [[ -e "$run_output" ]]; then
  echo "refusing to overwrite mastery output: $run_output" >&2
  exit 1
fi
IFS=, read -ra eval_devices <<<"$cuda_devices"
if [[ ${#eval_devices[@]} -ne $tensor_parallel_size ]]; then
  echo "CUDA device count must equal tensor parallel size" >&2
  exit 1
fi
for device in "${eval_devices[@]}"; do
  if [[ -n "$("$nvidia_smi_bin" --id="$device" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "refusing to launch while another GPU process is active on device $device" >&2
    exit 1
  fi
done
if [[ "$model" = /* ]] && [[ ! -f "$model/STABLE" ]]; then
  echo "local model has no STABLE marker: $model" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
export PRIME_API_KEY=${PRIME_API_KEY:-${PIT_KEY:-local-eval}}
if [[ "$dry_run" == true ]]; then
  serve_config=$(mktemp "${TMPDIR:-/tmp}/qwen35-27b-mastery-v2.XXXXXX")
  cleanup_preflight() {
    rm -f "$serve_config"
  }
  trap cleanup_preflight EXIT
else
  mkdir -p "$run_output"
  serve_config=$run_output/inference.toml
fi

cat >"$serve_config" <<EOF
backend_port = $backend_port

[server]
port = $router_port
liveness_timeout_seconds = 30.0

[vllm]
model = "$model"
revision = "$revision"
dtype = "auto"
max_model_len = 65536
enforce_eager = true
trust_remote_code = false
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"
tensor_parallel_size = $tensor_parallel_size
data_parallel_size = 1
data_parallel_rpc_port = $data_parallel_rpc_port
gpu_memory_utilization = 0.80
max_num_seqs = 4

[log]
level = "info"
vf_level = "info"
json_logging = false
log_data = false
interval = 10.0
EOF

if [[ "$dry_run" == true ]]; then
  "$inference_bin" @ "$serve_config" --dry-run true
  echo "baseline launch preflight passed: $run_output"
  exit 0
fi

inference_pid=
eval_pid=
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$eval_pid" ]]; then
    kill "$eval_pid" 2>/dev/null || true
    wait "$eval_pid" 2>/dev/null || true
  fi
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
  if curl -fsS "http://127.0.0.1:$backend_port/health" >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:$backend_port/health" >/dev/null; then
  echo "inference did not become healthy within 480 seconds; see $serve_log" >&2
  exit 1
fi

PRIME_MASTERY_OUTPUT_ROOT=$output_root \
  MODEL_REVISION=$revision \
  EVAL_CLIENT_BASE_URL="http://127.0.0.1:$backend_port/v1" \
  "$eval_driver" "$model" "$label" &
eval_pid=$!

completed_pid=
set +e
wait -n -p completed_pid "$inference_pid" "$eval_pid"
completed_status=$?
set -e
if [[ "$completed_pid" == "$inference_pid" ]]; then
  kill "$eval_pid" 2>/dev/null || true
  wait "$eval_pid" 2>/dev/null || true
  if ((completed_status == 0)); then
    completed_status=1
  fi
  echo "inference exited before the mastery evaluation completed; see $serve_log" >&2
  exit "$completed_status"
fi
if ((completed_status != 0)); then
  exit "$completed_status"
fi
echo "mastery baseline completed: $run_output"
