#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:?canonical R7 path is required}
evaluation_root=${PRIME_AGENT_RUNTIME_COMPARISON_ROOT:-/ephemeral/evals/qwen35-27b-prime-agent-runtime-v1/r1}
old_version=${OLD_PRIME_AGENT_VERSION:-0.7.2-beta.495.1.97b994c}
current_version=${CURRENT_PRIME_AGENT_VERSION:-0.7.3-beta.518.1.f8f0036}
old_config_dir=experiments/qwen35-27b-procedural-harness-master-v1
current_config_dir=$old_config_dir/current-runtime-v1
old_natural_config=$root/$old_config_dir/natural-policy-connectivity-bounded-gate.toml
old_action_config=$root/$old_config_dir/harness-action-admission.toml
current_natural_config=$root/$current_config_dir/natural-policy-connectivity-bounded-gate.toml
current_action_config=$root/$current_config_dir/harness-action-admission.toml
server_driver=$root/scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh
battery_driver=$root/scripts/run_qwen35_27b_natural_yield_sdpo_gate_battery_v1.sh
model_revision=${MODEL_REVISION:-8f0568faed72d0db2e2258c18b1aabdcefd680cc}
natural_start=${PRIME_AGENT_RUNTIME_NATURAL_START_INDEX:-3700000}
local_work_start=${PRIME_AGENT_RUNTIME_LOCAL_WORK_START_INDEX:-3700001}
state_start=${PRIME_AGENT_RUNTIME_STATE_START_INDEX:-3710000}
send_start=${PRIME_AGENT_RUNTIME_SEND_START_INDEX:-3711000}
source_commit=${EVALUATION_SOURCE_COMMIT:-$(git -C "$root" rev-parse HEAD)}
verifiers_commit=${VERIFIERS_EVALUATION_COMMIT:-$(git -C "$root/deps/verifiers" rev-parse HEAD)}

cd "$root"
for path in \
  "$old_natural_config" \
  "$old_action_config" \
  "$current_natural_config" \
  "$current_action_config"; do
  if [[ ! -f "$path" ]]; then
    echo "runtime comparison config is missing: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$model/STABLE" ]]; then
  echo "canonical model is not stable: $model" >&2
  exit 1
fi
if [[ -e "$evaluation_root" ]]; then
  echo "refusing to overwrite runtime comparison: $evaluation_root" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to evaluate while another GPU process is active" >&2
  exit 1
fi

PRIME_AGENT_VERSION=$old_version scripts/build_prime_agent_runtime_image_v1.sh >/dev/null
PRIME_AGENT_VERSION=$current_version scripts/build_prime_agent_runtime_image_v1.sh >/dev/null
old_image=rlm-prime-agent-runtime:$old_version-node22.19.0
current_image=rlm-prime-agent-runtime:$current_version-node22.19.0
nonblocking_prompt="For slow or independently completing work, use a nonblocking control loop"
if docker run --rm "$old_image" sh -lc \
  "grep -R -F -m1 '$nonblocking_prompt' /var/tmp/vf-prime-agent/$old_version/lib/node_modules/prime-agent/dist >/dev/null 2>&1"; then
  echo "historical Prime Agent image unexpectedly contains the new nonblocking prompt" >&2
  exit 1
fi
if ! docker run --rm "$current_image" sh -lc \
  "grep -R -F -m1 '$nonblocking_prompt' /var/tmp/vf-prime-agent/$current_version/lib/node_modules/prime-agent/dist >/dev/null 2>&1"; then
  echo "current Prime Agent image does not contain the expected nonblocking prompt" >&2
  exit 1
fi

mkdir -p "$evaluation_root"
{
  printf 'prime_rl_source_commit=%s\n' "$source_commit"
  printf 'verifiers_commit=%s\n' "$verifiers_commit"
  printf 'model=%s\n' "$model"
  printf 'model_revision=%s\n' "$model_revision"
  printf 'old_prime_agent_version=%s\n' "$old_version"
  printf 'current_prime_agent_version=%s\n' "$current_version"
  printf 'current_prime_agent_commit=f8f0036cc2da1a640aad990ae8dcb7c4820ce32e\n'
  printf 'nonblocking_prompt_commit=7ca44937f\n'
  printf 'old_runtime_image_id=%s\n' "$(docker image inspect --format='{{.Id}}' "$old_image")"
  printf 'current_runtime_image_id=%s\n' "$(docker image inspect --format='{{.Id}}' "$current_image")"
  printf 'natural_start_index=%s\n' "$natural_start"
  printf 'local_work_start_index=%s\n' "$local_work_start"
  printf 'state_start_index=%s\n' "$state_start"
  printf 'send_start_index=%s\n' "$send_start"
} >"$evaluation_root/VERSIONS.txt"

run_arm() {
  local label=$1
  local version=$2
  local config_dir=$3
  local natural_config=$4
  local action_config=$5
  local devices=$6
  local backend_port=$7
  local router_port=$8
  local rpc_port=$9
  PRIME_AGENT_VERSION=$version \
  PRIME_MASTERY_OUTPUT_ROOT=$evaluation_root \
  PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  EVAL_DRIVER=$battery_driver \
  EVAL_EXPERIMENT_DIR=$config_dir \
  EVAL_CUDA_VISIBLE_DEVICES=$devices \
  EVAL_TENSOR_PARALLEL_SIZE=4 \
  EVAL_BACKEND_PORT=$backend_port \
  EVAL_ROUTER_PORT=$router_port \
  EVAL_DATA_PARALLEL_RPC_PORT=$rpc_port \
  EVAL_VIRTUAL_MEMORY_LIMIT_KIB=${EVAL_VIRTUAL_MEMORY_LIMIT_KIB:-100663296} \
  NATURAL_POLICY_PROBE_CONFIG=$natural_config \
  HARNESS_ACTION_ADMISSION_CONFIG=$action_config \
  NATURAL_YIELD_GATE_START_INDEX=$natural_start \
  NATURAL_YIELD_LOCAL_WORK_GATE_START_INDEX=$local_work_start \
  NATURAL_YIELD_STATE_GATE_START_INDEX=$state_start \
  NATURAL_YIELD_SEND_GATE_START_INDEX=$send_start \
    "$server_driver" "$model" "$label" "$model_revision" \
    >"$evaluation_root/$label-driver.log" 2>&1
}

run_arm old-runtime "$old_version" "$old_config_dir" \
  "$old_natural_config" "$old_action_config" 0,1,2,3 8100 8000 13345 &
old_pid=$!
run_arm current-runtime "$current_version" "$current_config_dir" \
  "$current_natural_config" "$current_action_config" 4,5,6,7 8200 8001 13346 &
current_pid=$!
old_status=0
current_status=0
wait "$old_pid" || old_status=$?
wait "$current_pid" || current_status=$?
if ((old_status != 0 || current_status != 0)); then
  echo "runtime comparison failed: old=$old_status current=$current_status" >&2
  exit 1
fi

.venv/bin/python -m scripts.compare_prime_agent_runtime_natural_yield_v1 \
  "$evaluation_root" old-runtime current-runtime \
  --expected-base-version "$old_version" \
  --expected-candidate-version "$current_version" \
  --output "$evaluation_root/COMPARISON.json"

echo "Prime Agent runtime comparison completed: $evaluation_root"
