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
eval_driver=${EVAL_DRIVER:-scripts/run_qwen35_27b_prime_agent_mastery_battery_v2.sh}
eval_experiment=${EVAL_EXPERIMENT_DIR:-experiments/qwen35-27b-prime-agent-mastery-v2}
inference_bin=${INFERENCE_BIN:-$root/.venv/bin/inference}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
nvidia_smi_bin=${NVIDIA_SMI_BIN:-nvidia-smi}
router_bin=${VLLM_ROUTER_BIN:-$root/.venv/bin/vllm-router}
curl_bin=${CURL_BIN:-curl}
dry_run=${BASELINE_DRY_RUN:-false}
eval_dtype=${EVAL_DTYPE:-auto}
eval_max_model_len=${EVAL_MAX_MODEL_LEN:-65536}
eval_gpu_memory_utilization=${EVAL_GPU_MEMORY_UTILIZATION:-0.80}
eval_max_num_seqs=${EVAL_MAX_NUM_SEQS:-4}
eval_max_num_batched_tokens=${EVAL_MAX_NUM_BATCHED_TOKENS:-}
eval_language_model_only=${EVAL_LANGUAGE_MODEL_ONLY:-false}
eval_disable_custom_all_reduce=${EVAL_DISABLE_CUSTOM_ALL_REDUCE:-false}
eval_xdg_config_home=${EVAL_XDG_CONFIG_HOME:-$run_output/.config}
eval_served_model=${EVAL_SERVED_MODEL:-$model}
eval_lora_path=${EVAL_LORA_PATH:-}
eval_lora_name=${EVAL_LORA_NAME:-}
eval_max_lora_rank=${EVAL_MAX_LORA_RANK:-16}
eval_python_bin=${EVAL_PYTHON_BIN:-$root/.venv/bin/python}

cd "$root"
export PATH="$(dirname "$router_bin"):$root/.venv/bin:$PATH"
if [[ ! -x "$inference_bin" || ! -x "$eval_bin" || ! -x "$eval_python_bin" ]]; then
  echo "Prime-RL inference/eval executables are missing" >&2
  exit 1
fi
if [[ ! -x "$router_bin" ]]; then
  echo "vllm-router is missing; install Prime-RL with its disagg or all extras" >&2
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
if [[ ! "$eval_max_model_len" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAX_MODEL_LEN must be a positive integer" >&2
  exit 1
fi
if [[ ! "$eval_max_num_seqs" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAX_NUM_SEQS must be a positive integer" >&2
  exit 1
fi
if [[ -n "$eval_max_num_batched_tokens" ]] \
  && [[ ! "$eval_max_num_batched_tokens" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAX_NUM_BATCHED_TOKENS must be empty or a positive integer" >&2
  exit 1
fi
if [[ ! "$eval_gpu_memory_utilization" =~ ^0\.[0-9]+$ ]]; then
  echo "EVAL_GPU_MEMORY_UTILIZATION must be a decimal between zero and one" >&2
  exit 1
fi
case "$eval_dtype" in
  auto|bfloat16|float16|float32) ;;
  *) echo "unsupported EVAL_DTYPE: $eval_dtype" >&2; exit 1 ;;
esac
case "$eval_language_model_only" in
  true|false) ;;
  *) echo "EVAL_LANGUAGE_MODEL_ONLY must be true or false" >&2; exit 1 ;;
esac
case "$eval_disable_custom_all_reduce" in
  true|false) ;;
  *) echo "EVAL_DISABLE_CUSTOM_ALL_REDUCE must be true or false" >&2; exit 1 ;;
esac
if [[ -n "$eval_lora_path" || -n "$eval_lora_name" ]]; then
  if [[ -z "$eval_lora_path" || -z "$eval_lora_name" ]]; then
    echo "EVAL_LORA_PATH and EVAL_LORA_NAME must be set together" >&2
    exit 1
  fi
  if [[ ! "$eval_max_lora_rank" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_MAX_LORA_RANK must be a positive integer" >&2
    exit 1
  fi
  if [[ ! -f "$eval_lora_path/adapter_config.json" \
    || ! -f "$eval_lora_path/adapter_model.safetensors" ]]; then
    echo "LoRA adapter is incomplete: $eval_lora_path" >&2
    exit 1
  fi
fi
mapfile -t prime_agent_versions < <(
  awk -F'"' '/^version = / {print $2}' \
    "$root"/"$eval_experiment"/*.toml \
    | sort -u
)
if [[ ${#prime_agent_versions[@]} -ne 1 ]]; then
  echo "mastery configs must pin exactly one Prime Agent version" >&2
  exit 1
fi
prime_agent_version=${prime_agent_versions[0]}
prime_agent_release_base=https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev
if ! checksums=$("$curl_bin" -fsSL \
  "$prime_agent_release_base/releases/v$prime_agent_version/SHA256SUMS"); then
  echo "Prime Agent artifact is unavailable: $prime_agent_version" >&2
  exit 1
fi
if ! grep -Eq "[[:space:]]prime-agent-$prime_agent_version\\.tgz$" <<<"$checksums"; then
  echo "Prime Agent checksum manifest lacks its package: $prime_agent_version" >&2
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
  preflight_dir=$(mktemp -d "${TMPDIR:-/tmp}/qwen35-27b-mastery-v2.XXXXXX")
  serve_config=$preflight_dir/inference.toml
  cleanup_preflight() {
    rm -f "$serve_config"
    rmdir "$preflight_dir"
  }
  trap cleanup_preflight EXIT
else
  mkdir -p "$run_output"
  mkdir -p "$eval_xdg_config_home"
  export XDG_CONFIG_HOME=$eval_xdg_config_home
  serve_config=$run_output/inference.toml
fi

max_num_batched_tokens_line=
if [[ -n "$eval_max_num_batched_tokens" ]]; then
  max_num_batched_tokens_line="max_num_batched_tokens = $eval_max_num_batched_tokens"
fi
lora_config_lines=
if [[ -n "$eval_lora_path" ]]; then
  lora_config_lines=$(printf 'enable_lora = true\nmax_loras = 1\nmax_cpu_loras = 1\nmax_lora_rank = %s' \
    "$eval_max_lora_rank")
fi

cat >"$serve_config" <<EOF
backend_port = $backend_port

[server]
port = $router_port
liveness_timeout_seconds = 30.0

[vllm]
model = "$model"
revision = "$revision"
dtype = "$eval_dtype"
max_model_len = $eval_max_model_len
language_model_only = $eval_language_model_only
enforce_eager = true
trust_remote_code = false
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"
tensor_parallel_size = $tensor_parallel_size
disable_custom_all_reduce = $eval_disable_custom_all_reduce
data_parallel_size = 1
data_parallel_rpc_port = $data_parallel_rpc_port
gpu_memory_utilization = $eval_gpu_memory_utilization
max_num_seqs = $eval_max_num_seqs
$max_num_batched_tokens_line
$lora_config_lines

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
health_pid=
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$health_pid" ]]; then
    kill "$health_pid" 2>/dev/null || true
    wait "$health_pid" 2>/dev/null || true
  fi
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
  if "$curl_bin" -fsS "http://127.0.0.1:$backend_port/health" >/dev/null; then
    break
  fi
  sleep 1
done
if ! "$curl_bin" -fsS "http://127.0.0.1:$backend_port/health" >/dev/null; then
  echo "inference did not become healthy within 480 seconds; see $serve_log" >&2
  exit 1
fi
if [[ -n "$eval_lora_path" ]]; then
  lora_payload=$("$eval_python_bin" - "$eval_lora_name" "$eval_lora_path" <<'PY'
import json
import sys

print(json.dumps({"lora_name": sys.argv[1], "lora_path": sys.argv[2]}))
PY
)
  if ! "$curl_bin" -fsS -X POST \
    -H 'Content-Type: application/json' \
    --data-binary "$lora_payload" \
    "http://127.0.0.1:$backend_port/load_lora_adapter" >/dev/null; then
    echo "failed to load LoRA adapter $eval_lora_name from $eval_lora_path" >&2
    exit 1
  fi
fi

PRIME_MASTERY_OUTPUT_ROOT=$output_root \
  MODEL_REVISION=$revision \
  EVAL_BASE_MODEL=$model \
  EVAL_LORA_NAME=$eval_lora_name \
  EVAL_LORA_PATH=$eval_lora_path \
  EVAL_LORA_RANK=$eval_max_lora_rank \
  EVAL_CLIENT_BASE_URL="http://127.0.0.1:$backend_port/v1" \
  "$eval_driver" "$eval_served_model" "$label" &
eval_pid=$!

# The Prime-RL inference parent can remain alive after its vLLM EngineCore dies.
# Watch the serving endpoint independently so a dead child cannot contaminate
# subsequent draws with a cascade of provider-error records.
(
  failures=0
  while true; do
    if "$curl_bin" -fsS "http://127.0.0.1:$backend_port/health" >/dev/null; then
      failures=0
    else
      failures=$((failures + 1))
      if ((failures >= 3)); then
        exit 1
      fi
    fi
    sleep 2
  done
) &
health_pid=$!

completed_pid=
set +e
wait -n -p completed_pid "$inference_pid" "$eval_pid" "$health_pid"
completed_status=$?
set -e
if [[ "$completed_pid" == "$inference_pid" || "$completed_pid" == "$health_pid" ]]; then
  kill "$eval_pid" 2>/dev/null || true
  wait "$eval_pid" 2>/dev/null || true
  if ((completed_status == 0)); then
    completed_status=1
  fi
  echo "inference exited or became unhealthy before the mastery evaluation completed; see $serve_log" >&2
  exit "$completed_status"
fi
if ((completed_status != 0)); then
  exit "$completed_status"
fi
echo "mastery baseline completed: $run_output"
