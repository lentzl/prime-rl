#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
child_model=${2:?child model path required}
label=${3:?evaluation label required}
revision=${4:?model revision required}
output_root=${QWEN38_QUALIFICATION_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-spade-dual-dense-v1}
run_output=$output_root/$label
eval_driver=${EVAL_DRIVER:-scripts/run_qwen38_27b_prime_harness_qualification_v1.sh}
inference_bin=${INFERENCE_BIN:-$root/.venv/bin/inference}
inference_dir=$(cd "$(dirname "$inference_bin")" && pwd)
export PATH="$inference_dir:$PATH"
uv_bin=${UV_BIN:-$(command -v uv || true)}
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin=$HOME/.local/bin/uv
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv executable not found" >&2
  exit 1
fi
external_model=${DUAL_EXTERNAL_MODEL:-q35-2b-dual-policy}
coordinator_backend_port=${COORDINATOR_BACKEND_PORT:-8101}
child_backend_port=${CHILD_BACKEND_PORT:-8102}
proxy_port=${DUAL_PROXY_PORT:-8100}
leak_coordinator_exact_action=${DUAL_LEAK_COORDINATOR_EXACT_ACTION:-0}
leak_document_manager_exact_action=${DUAL_LEAK_DOCUMENT_MANAGER_EXACT_ACTION:-0}
leak_coordinator_return_action=${DUAL_LEAK_COORDINATOR_RETURN_ACTION:-0}
typed_coordinator_return=${DUAL_TYPED_COORDINATOR_RETURN:-0}
root_coordinator_contract=${DUAL_ROOT_COORDINATOR_CONTRACT:-0}
document_root_utility_decision_contract=${DUAL_DOCUMENT_ROOT_UTILITY_DECISION_CONTRACT:-0}
document_root_causal_utility_decision_contract=${DUAL_DOCUMENT_ROOT_CAUSAL_UTILITY_DECISION_CONTRACT:-0}
leaf_reporter_contract=${DUAL_LEAF_REPORTER_CONTRACT:-0}
leaf_inline_evidence=${DUAL_LEAF_INLINE_EVIDENCE:-0}
leaf_compute_report_scaffold=${DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD:-0}
document_leaf_compute_report_scaffold=${DUAL_DOCUMENT_LEAF_COMPUTE_REPORT_SCAFFOLD:-0}
document_manager_fanin_scaffold=${DUAL_DOCUMENT_MANAGER_FANIN_SCAFFOLD:-0}
document_manager_wait_scaffold=${DUAL_DOCUMENT_MANAGER_WAIT_SCAFFOLD:-0}
document_manager_termination_scaffold=${DUAL_DOCUMENT_MANAGER_TERMINATION_SCAFFOLD:-0}
document_root_report_relay_scaffold=${DUAL_DOCUMENT_ROOT_REPORT_RELAY_SCAFFOLD:-0}
document_root_topology_normalization_scaffold=${DUAL_DOCUMENT_ROOT_TOPOLOGY_NORMALIZATION_SCAFFOLD:-0}
document_root_flat_fanin_scaffold=${DUAL_DOCUMENT_ROOT_FLAT_FANIN_SCAFFOLD:-0}
typed_child_report=${DUAL_TYPED_CHILD_REPORT:-0}
child_authored_compute=${DUAL_CHILD_AUTHORED_COMPUTE:-0}
depth_default_child=${DUAL_DEPTH_DEFAULT_CHILD:-0}
scaffold_profile=${DUAL_SCAFFOLD_PROFILE:-custom}

cd "$root"
for model in "$coordinator_model" "$child_model"; do
  if [[ "$model" != /* || ! -f "$model/STABLE" || ! -f "$model/model.safetensors" ]]; then
    echo "dense role model is not an absolute complete checkpoint: $model" >&2
    exit 1
  fi
done
if [[ -e "$run_output" ]]; then
  echo "refusing to overwrite dual-policy evaluation: $run_output" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch dual-policy evaluation while a GPU process is active" >&2
  exit 1
fi
if [[ "$leak_coordinator_return_action" != 0 && "$leak_coordinator_return_action" != 1 ]]; then
  echo "DUAL_LEAK_COORDINATOR_RETURN_ACTION must be 0 or 1" >&2
  exit 1
fi
if [[ "$leak_coordinator_exact_action" != 0 && "$leak_coordinator_exact_action" != 1 ]]; then
  echo "DUAL_LEAK_COORDINATOR_EXACT_ACTION must be 0 or 1" >&2
  exit 1
fi
if [[ "$leak_document_manager_exact_action" != 0 && "$leak_document_manager_exact_action" != 1 ]]; then
  echo "DUAL_LEAK_DOCUMENT_MANAGER_EXACT_ACTION must be 0 or 1" >&2
  exit 1
fi
if [[ "$typed_coordinator_return" != 0 && "$typed_coordinator_return" != 1 ]]; then
  echo "DUAL_TYPED_COORDINATOR_RETURN must be 0 or 1" >&2
  exit 1
fi
if [[ "$root_coordinator_contract" != 0 && "$root_coordinator_contract" != 1 ]]; then
  echo "DUAL_ROOT_COORDINATOR_CONTRACT must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_root_utility_decision_contract" != 0 && "$document_root_utility_decision_contract" != 1 ]]; then
  echo "DUAL_DOCUMENT_ROOT_UTILITY_DECISION_CONTRACT must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_root_causal_utility_decision_contract" != 0 && "$document_root_causal_utility_decision_contract" != 1 ]]; then
  echo "DUAL_DOCUMENT_ROOT_CAUSAL_UTILITY_DECISION_CONTRACT must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_root_utility_decision_contract" == 1 && "$document_root_causal_utility_decision_contract" == 1 ]]; then
  echo "historical and causal document utility contracts are mutually exclusive" >&2
  exit 1
fi
if [[ "$leaf_reporter_contract" != 0 && "$leaf_reporter_contract" != 1 ]]; then
  echo "DUAL_LEAF_REPORTER_CONTRACT must be 0 or 1" >&2
  exit 1
fi
if [[ "$typed_child_report" != 0 && "$typed_child_report" != 1 ]]; then
  echo "DUAL_TYPED_CHILD_REPORT must be 0 or 1" >&2
  exit 1
fi
if [[ "$child_authored_compute" != 0 && "$child_authored_compute" != 1 ]]; then
  echo "DUAL_CHILD_AUTHORED_COMPUTE must be 0 or 1" >&2
  exit 1
fi
if [[ "$depth_default_child" != 0 && "$depth_default_child" != 1 ]]; then
  echo "DUAL_DEPTH_DEFAULT_CHILD must be 0 or 1" >&2
  exit 1
fi
if [[ "$leaf_inline_evidence" != 0 && "$leaf_inline_evidence" != 1 ]]; then
  echo "DUAL_LEAF_INLINE_EVIDENCE must be 0 or 1" >&2
  exit 1
fi
if [[ "$leaf_compute_report_scaffold" != 0 && "$leaf_compute_report_scaffold" != 1 ]]; then
  echo "DUAL_LEAF_COMPUTE_REPORT_SCAFFOLD must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_leaf_compute_report_scaffold" != 0 && "$document_leaf_compute_report_scaffold" != 1 ]]; then
  echo "DUAL_DOCUMENT_LEAF_COMPUTE_REPORT_SCAFFOLD must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_manager_fanin_scaffold" != 0 && "$document_manager_fanin_scaffold" != 1 ]]; then
  echo "DUAL_DOCUMENT_MANAGER_FANIN_SCAFFOLD must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_manager_wait_scaffold" != 0 && "$document_manager_wait_scaffold" != 1 ]]; then
  echo "DUAL_DOCUMENT_MANAGER_WAIT_SCAFFOLD must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_manager_termination_scaffold" != 0 && "$document_manager_termination_scaffold" != 1 ]]; then
  echo "DUAL_DOCUMENT_MANAGER_TERMINATION_SCAFFOLD must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_root_report_relay_scaffold" != 0 && "$document_root_report_relay_scaffold" != 1 ]]; then
  echo "DUAL_DOCUMENT_ROOT_REPORT_RELAY_SCAFFOLD must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_root_topology_normalization_scaffold" != 0 && "$document_root_topology_normalization_scaffold" != 1 ]]; then
  echo "DUAL_DOCUMENT_ROOT_TOPOLOGY_NORMALIZATION_SCAFFOLD must be 0 or 1" >&2
  exit 1
fi
if [[ "$document_root_topology_normalization_scaffold" == 1 && "$leak_coordinator_exact_action" == 1 ]]; then
  echo "document root topology normalization and exact coordinator action are mutually exclusive" >&2
  exit 1
fi
if [[ "$document_root_flat_fanin_scaffold" != 0 && "$document_root_flat_fanin_scaffold" != 1 ]]; then
  echo "DUAL_DOCUMENT_ROOT_FLAT_FANIN_SCAFFOLD must be 0 or 1" >&2
  exit 1
fi
if [[ "$leak_coordinator_return_action" == 1 && "$typed_coordinator_return" == 1 ]]; then
  echo "exact and typed coordinator-return scaffolds are mutually exclusive" >&2
  exit 1
fi
if [[ "$leaf_compute_report_scaffold" == 1 && "$typed_child_report" == 1 ]]; then
  echo "leaf compute-report and typed child-report scaffolds are mutually exclusive" >&2
  exit 1
fi
if [[ "$leaf_compute_report_scaffold" == 1 && "$leaf_inline_evidence" != 1 ]]; then
  echo "leaf compute-report scaffold requires DUAL_LEAF_INLINE_EVIDENCE=1" >&2
  exit 1
fi
if [[ "$child_authored_compute" == 1 && "$typed_child_report" != 1 ]]; then
  echo "child-authored compute requires DUAL_TYPED_CHILD_REPORT=1" >&2
  exit 1
fi
case "$scaffold_profile" in
  custom) ;;
  tight_answer_free_child_reporting_v1)
    if [[ "$leak_coordinator_exact_action" != 0 \
      || "$leak_coordinator_return_action" != 0 \
      || "$typed_coordinator_return" != 0 \
      || "$root_coordinator_contract" != 1 \
      || "$leaf_reporter_contract" != 1 \
      || "$leaf_inline_evidence" != 1 \
      || "$leaf_compute_report_scaffold" != 0 \
      || "$typed_child_report" != 1 \
      || "$child_authored_compute" != 0 ]]; then
      echo "tight_answer_free_child_reporting_v1 scaffold flags do not match its frozen contract" >&2
      exit 1
    fi
    ;;
  tight_learned_semantic_probe_v1)
    if [[ "$leak_coordinator_exact_action" != 0 \
      || "$leak_coordinator_return_action" != 0 \
      || "$typed_coordinator_return" != 0 \
      || "$root_coordinator_contract" != 1 \
      || "$leaf_reporter_contract" != 1 \
      || "$leaf_inline_evidence" != 1 \
      || "$leaf_compute_report_scaffold" != 0 \
      || "$typed_child_report" != 1 \
      || "$child_authored_compute" != 1 ]]; then
      echo "tight_learned_semantic_probe_v1 scaffold flags do not match its frozen contract" >&2
      exit 1
    fi
    ;;
  *) echo "unsupported dual-policy scaffold profile: $scaffold_profile" >&2; exit 1 ;;
esac
mkdir -p "$run_output"

write_inference_config() {
  local path=$1
  local model=$2
  local backend_port=$3
  local router_port=$4
  local rpc_port=$5
  cat >"$path" <<EOF
backend_port = $backend_port

[server]
port = $router_port
liveness_timeout_seconds = 30.0

[vllm]
model = "$model"
revision = "$revision"
dtype = "bfloat16"
max_model_len = 32768
language_model_only = true
enforce_eager = true
trust_remote_code = false
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"
tensor_parallel_size = 1
disable_custom_all_reduce = false
data_parallel_size = 1
data_parallel_rpc_port = $rpc_port
gpu_memory_utilization = 0.80
max_num_seqs = 8
max_num_batched_tokens = 4096

[log]
level = "info"
vf_level = "info"
json_logging = false
log_data = false
interval = 10.0
EOF
}

coordinator_config=$run_output/coordinator-inference.toml
child_config=$run_output/child-inference.toml
routing_audit=$run_output/ROUTING_AUDIT.jsonl
write_inference_config "$coordinator_config" "$coordinator_model" "$coordinator_backend_port" 8001 13346
write_inference_config "$child_config" "$child_model" "$child_backend_port" 8002 13347

coordinator_pid=
child_pid=
proxy_pid=
eval_pid=
monitor_pid=
cleanup() {
  trap - EXIT INT TERM
  for pid in "$monitor_pid" "$eval_pid" "$proxy_pid" "$child_pid" "$coordinator_pid"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 "$inference_bin" @ "$coordinator_config" \
  >"$run_output/coordinator-inference.log" 2>&1 &
coordinator_pid=$!
CUDA_VISIBLE_DEVICES=1 "$inference_bin" @ "$child_config" \
  >"$run_output/child-inference.log" 2>&1 &
child_pid=$!

for _ in $(seq 1 480); do
  if ! kill -0 "$coordinator_pid" 2>/dev/null || ! kill -0 "$child_pid" 2>/dev/null; then
    echo "a role inference process exited during startup" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$coordinator_backend_port/health" >/dev/null \
    && curl -fsS "http://127.0.0.1:$child_backend_port/health" >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:$coordinator_backend_port/health" >/dev/null \
  || ! curl -fsS "http://127.0.0.1:$child_backend_port/health" >/dev/null; then
  echo "dual role inference did not become healthy" >&2
  exit 1
fi

proxy_args=(
  --port "$proxy_port"
  --coordinator-url "http://127.0.0.1:$coordinator_backend_port/v1"
  --coordinator-model "$coordinator_model"
  --child-url "http://127.0.0.1:$child_backend_port/v1"
  --child-model "$child_model"
  --external-model "$external_model"
  --audit-log "$routing_audit"
)
if [[ "$leak_coordinator_exact_action" == 1 ]]; then
  proxy_args+=(--leak-coordinator-exact-action)
fi
if [[ "$leak_document_manager_exact_action" == 1 ]]; then
  proxy_args+=(--leak-document-manager-exact-action)
fi
if [[ "$leak_coordinator_return_action" == 1 ]]; then
  proxy_args+=(--leak-coordinator-return-action)
fi
if [[ "$typed_coordinator_return" == 1 ]]; then
  proxy_args+=(--typed-coordinator-return)
fi
if [[ "$root_coordinator_contract" == 1 ]]; then
  proxy_args+=(--root-coordinator-contract)
fi
if [[ "$document_root_utility_decision_contract" == 1 ]]; then
  proxy_args+=(--document-root-utility-decision-contract)
fi
if [[ "$document_root_causal_utility_decision_contract" == 1 ]]; then
  proxy_args+=(--document-root-causal-utility-decision-contract)
fi
if [[ "$leaf_reporter_contract" == 1 ]]; then
  proxy_args+=(--leaf-reporter-contract)
fi
if [[ "$leaf_inline_evidence" == 1 ]]; then
  proxy_args+=(--leaf-inline-evidence)
fi
if [[ "$leaf_compute_report_scaffold" == 1 ]]; then
  proxy_args+=(--leaf-compute-report-scaffold)
fi
if [[ "$document_leaf_compute_report_scaffold" == 1 ]]; then
  proxy_args+=(--document-leaf-compute-report-scaffold)
fi
if [[ "$document_manager_fanin_scaffold" == 1 ]]; then
  proxy_args+=(--document-manager-fanin-scaffold)
fi
if [[ "$document_manager_wait_scaffold" == 1 ]]; then
  proxy_args+=(--document-manager-wait-scaffold)
fi
if [[ "$document_manager_termination_scaffold" == 1 ]]; then
  proxy_args+=(--document-manager-termination-scaffold)
fi
if [[ "$document_root_report_relay_scaffold" == 1 ]]; then
  proxy_args+=(--document-root-report-relay-scaffold)
fi
if [[ "$document_root_topology_normalization_scaffold" == 1 ]]; then
  proxy_args+=(--document-root-topology-normalization-scaffold)
fi
if [[ "$document_root_flat_fanin_scaffold" == 1 ]]; then
  proxy_args+=(--document-root-flat-fanin-scaffold)
fi
if [[ "$typed_child_report" == 1 ]]; then
  proxy_args+=(--typed-child-report)
fi
if [[ "$child_authored_compute" == 1 ]]; then
  proxy_args+=(--child-authored-compute)
fi
if [[ "$depth_default_child" == 1 ]]; then
  proxy_args+=(--depth-default-child)
fi

"$uv_bin" run --no-sync scripts/dual_policy_openai_proxy_v1.py \
  "${proxy_args[@]}" \
  >"$run_output/proxy.log" 2>&1 &
proxy_pid=$!
for _ in $(seq 1 60); do
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "dual-policy proxy exited during startup" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$proxy_port/health" >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -fsS "http://127.0.0.1:$proxy_port/health" >/dev/null; then
  echo "dual-policy proxy did not become healthy" >&2
  exit 1
fi

MODEL_REVISION="$revision" \
EVAL_CLIENT_BASE_URL="http://127.0.0.1:$proxy_port/v1" \
"$eval_driver" "$external_model" "$label" &
eval_pid=$!

(
  failures=0
  while kill -0 "$eval_pid" 2>/dev/null; do
    for service_pid in "$coordinator_pid" "$child_pid" "$proxy_pid"; do
      if ! kill -0 "$service_pid" 2>/dev/null; then
        kill -TERM "$eval_pid" 2>/dev/null || true
        exit 1
      fi
    done
    if curl -fsS "http://127.0.0.1:$proxy_port/health" >/dev/null; then
      failures=0
    else
      failures=$((failures + 1))
      if ((failures >= 3)); then
        kill -TERM "$eval_pid" 2>/dev/null || true
        exit 1
      fi
    fi
    sleep 2
  done
) &
monitor_pid=$!

set +e
wait "$eval_pid"
eval_status=$?
kill "$monitor_pid" 2>/dev/null
wait "$monitor_pid" 2>/dev/null
set -e
if [[ $eval_status -ne 0 ]]; then
  echo "dual-policy service exited or became unhealthy before evaluation completed" >&2
  exit 1
fi

coordinator_sha=$(sha256sum "$coordinator_model/model.safetensors" | awk '{print $1}')
child_sha=$(sha256sum "$child_model/model.safetensors" | awk '{print $1}')
{
  printf 'dual_policy_scaffold_profile=%s\n' "$scaffold_profile"
  printf 'dual_leak_coordinator_exact_action=%s\n' "$leak_coordinator_exact_action"
  printf 'dual_leak_document_manager_exact_action=%s\n' "$leak_document_manager_exact_action"
  printf 'dual_leak_coordinator_return_action=%s\n' "$leak_coordinator_return_action"
  printf 'dual_typed_coordinator_return=%s\n' "$typed_coordinator_return"
  printf 'dual_root_coordinator_contract=%s\n' "$root_coordinator_contract"
  printf 'dual_document_root_utility_decision_contract=%s\n' "$document_root_utility_decision_contract"
  printf 'dual_document_root_causal_utility_decision_contract=%s\n' "$document_root_causal_utility_decision_contract"
  printf 'dual_leaf_reporter_contract=%s\n' "$leaf_reporter_contract"
  printf 'dual_leaf_inline_evidence=%s\n' "$leaf_inline_evidence"
  printf 'dual_leaf_compute_report_scaffold=%s\n' "$leaf_compute_report_scaffold"
  printf 'dual_document_leaf_compute_report_scaffold=%s\n' "$document_leaf_compute_report_scaffold"
  printf 'dual_document_manager_fanin_scaffold=%s\n' "$document_manager_fanin_scaffold"
  printf 'dual_document_manager_wait_scaffold=%s\n' "$document_manager_wait_scaffold"
  printf 'dual_document_manager_termination_scaffold=%s\n' "$document_manager_termination_scaffold"
  printf 'dual_document_root_report_relay_scaffold=%s\n' "$document_root_report_relay_scaffold"
  printf 'dual_document_root_topology_normalization_scaffold=%s\n' "$document_root_topology_normalization_scaffold"
  printf 'dual_document_root_flat_fanin_scaffold=%s\n' "$document_root_flat_fanin_scaffold"
  printf 'dual_typed_child_report=%s\n' "$typed_child_report"
  printf 'dual_child_authored_compute=%s\n' "$child_authored_compute"
  printf 'dual_depth_default_child=%s\n' "$depth_default_child"
  printf 'dual_policy_external_model=%s\n' "$external_model"
  printf 'coordinator_model_path=%s\n' "$coordinator_model"
  printf 'coordinator_model_sha256=%s\n' "$coordinator_sha"
  printf 'child_model_path=%s\n' "$child_model"
  printf 'child_model_sha256=%s\n' "$child_sha"
  sha256sum "$routing_audit" "$coordinator_config" "$child_config"
} >>"$run_output/VERSIONS.txt"
echo "dual-policy mastery evaluation completed: $run_output"
