#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base_model=${1:?canonical R7 path is required}
cumulative_run=${2:?cumulative SDPO run directory is required}
evaluation_root=${NATURAL_YIELD_CUMULATIVE_TARGET_ROOT:-/ephemeral/evals/qwen35-27b-natural-yield-sdpo-v1/cumulative-r1-targets}
server_driver=$root/scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh
target_driver=$root/scripts/run_qwen35_27b_natural_policy_connectivity_probe_v1.sh
target_config=$root/experiments/qwen35-27b-procedural-harness-master-v1/natural-policy-connectivity-bounded-gate.toml
model_revision=${MODEL_REVISION:-8f0568faed72d0db2e2258c18b1aabdcefd680cc}
target_start=${NATURAL_YIELD_GATE_START_INDEX:-3500000}
source_commit=${EVALUATION_SOURCE_COMMIT:-$(git -C "$root" rev-parse HEAD)}
verifiers_commit=${VERIFIERS_EVALUATION_COMMIT:-$(git -C "$root/deps/verifiers" rev-parse HEAD)}

cd "$root"
declare -A models=(
  [r7]="$base_model"
  [step1]="$cumulative_run/weights/step_1"
  [step2]="$cumulative_run/weights/step_2"
  [step4]="$cumulative_run/weights/step_4"
)
for label in r7 step1 step2 step4; do
  if [[ ! -f "${models[$label]}/STABLE" ]]; then
    echo "cumulative target model is not stable: ${models[$label]}" >&2
    exit 1
  fi
done
if [[ ! -f "$cumulative_run/CUMULATIVE_UPDATE.json" ]]; then
  echo "cumulative update validator report is absent" >&2
  exit 1
fi
if [[ -e "$evaluation_root" ]]; then
  echo "refusing to overwrite cumulative target gates: $evaluation_root" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to evaluate while another GPU process is active" >&2
  exit 1
fi

mkdir -p "$evaluation_root"
{
  printf 'prime_rl_source_commit=%s\n' "$source_commit"
  printf 'verifiers_commit=%s\n' "$verifiers_commit"
  printf 'model_revision=%s\n' "$model_revision"
  printf 'target_start_index=%s\n' "$target_start"
  for label in r7 step1 step2 step4; do
    printf '%s_model=%s\n' "$label" "${models[$label]}"
  done
} >"$evaluation_root/VERSIONS.txt"

run_target() {
  local label=$1
  local devices=$2
  local backend_port=$3
  local router_port=$4
  local rpc_port=$5
  local run_label=$label-natural-yield
  PRIME_MASTERY_OUTPUT_ROOT=$evaluation_root \
  PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  EVAL_DRIVER=$target_driver \
  EVAL_EXPERIMENT_DIR=experiments/qwen35-27b-procedural-harness-master-v1 \
  EVAL_CUDA_VISIBLE_DEVICES=$devices \
  EVAL_TENSOR_PARALLEL_SIZE=4 \
  EVAL_BACKEND_PORT=$backend_port \
  EVAL_ROUTER_PORT=$router_port \
  EVAL_DATA_PARALLEL_RPC_PORT=$rpc_port \
  EVAL_VIRTUAL_MEMORY_LIMIT_KIB=${EVAL_VIRTUAL_MEMORY_LIMIT_KIB:-100663296} \
  NATURAL_POLICY_PROBE_CONFIG=$target_config \
  NATURAL_POLICY_RUNG=natural_n1 \
  NATURAL_POLICY_PROBE_START_INDEX=$target_start \
    "$server_driver" "${models[$label]}" "$run_label" "$model_revision" \
    >"$evaluation_root/$label-driver.log" 2>&1
}

run_pair() {
  local left=$1
  local right=$2
  run_target "$left" 0,1,2,3 8100 8000 13345 &
  local left_pid=$!
  run_target "$right" 4,5,6,7 8200 8001 13346 &
  local right_pid=$!
  local left_status=0
  local right_status=0
  wait "$left_pid" || left_status=$?
  wait "$right_pid" || right_status=$?
  if ((left_status != 0 || right_status != 0)); then
    echo "target pair failed: $left=$left_status $right=$right_status" >&2
    return 1
  fi
}

run_pair r7 step1
run_pair step2 step4

for label in step1 step2 step4; do
  .venv/bin/python -m scripts.compare_natural_yield_sdpo_target_gates_v1 \
    "$evaluation_root" r7 "$label" \
    --output "$evaluation_root/COMPARISON-$label.json"
done
echo "cumulative natural-yield target gates completed: $evaluation_root"
