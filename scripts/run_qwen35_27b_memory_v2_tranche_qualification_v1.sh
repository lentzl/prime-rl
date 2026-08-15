#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_output=${MEMORY_TRANCHE_OUTPUT:-/ephemeral/subagent-rung/outputs/346-qwen35-27b-memory-v2-hybrid-tranche-v2}
output_root=${MEMORY_QUALIFICATION_OUTPUT:-/ephemeral/subagent-rung/evals/349-qwen35-27b-memory-v2-tranche-qualification-v1}
base_revision=fc05daec18b0a78c049392ed2e771dde82bdf654
steps=(1 2 4 8)
parallelism=${MEMORY_QUALIFICATION_PARALLELISM:-2}
device_groups=(0,1,2,3 4,5,6,7)
model_launcher=${MEMORY_QUALIFICATION_MODEL_LAUNCHER:-scripts/run_qwen35_27b_mastery_fast_screen_model_v1.sh}
service_port_stride=200

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to qualify while another GPU process is active" >&2
  exit 1
fi
if (( parallelism < 1 || parallelism > ${#device_groups[@]} )); then
  echo "MEMORY_QUALIFICATION_PARALLELISM must be 1 or ${#device_groups[@]}" >&2
  exit 1
fi
declare -A reserved_ports=()
for ((slot = 0; slot < parallelism; slot++)); do
  slot_ports=(
    "$((8000 + service_port_stride * slot))"
    "$((8100 + service_port_stride * slot))"
    "$((13345 + 100 * slot))"
  )
  for port in "${slot_ports[@]}"; do
    if [[ -n "${reserved_ports[$port]+present}" ]]; then
      echo "qualification slots must use distinct ports: $port" >&2
      exit 1
    fi
    reserved_ports[$port]=1
  done
done
for step in "${steps[@]}"; do
  checkpoint=$source_output/weights/step_$step
  if [[ ! -f "$checkpoint/STABLE" ]]; then
    echo "selected checkpoint is not stable: $checkpoint" >&2
    exit 1
  fi
done

models=(
  Qwen/Qwen3.5-27B
  "$source_output/weights/step_1"
  "$source_output/weights/step_2"
  "$source_output/weights/step_4"
  "$source_output/weights/step_8"
)
labels=(base step-1 step-2 step-4 step-8)
revisions=("$base_revision" "" "" "" "")
pending=()

for index in "${!models[@]}"; do
  run_output=$output_root/${labels[$index]}
  if [[ -f "$run_output/QUALIFICATION_COMPLETE" ]]; then
    echo "skipping completed qualification: ${labels[$index]}"
    continue
  fi
  if [[ -e "$run_output" ]]; then
    echo "refusing to mix partial qualification output: $run_output" >&2
    exit 1
  fi
  pending+=("$index")
done

active_pids=()
active_labels=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${active_pids[@]}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for pid in "${active_pids[@]}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

launch_slot() {
  local slot=$1
  local index=$2
  local backend_port=$((8100 + service_port_stride * slot))
  local router_port=$((8000 + service_port_stride * slot))
  local rpc_port=$((13345 + 100 * slot))

  echo "launching qualification ${labels[$index]} on GPUs ${device_groups[$slot]}"
  PRIME_MASTERY_OUTPUT_ROOT=$output_root \
    MASTERY_EVAL_DRIVER=scripts/run_qwen35_27b_memory_v2_combined_qualification.sh \
    EVAL_CUDA_VISIBLE_DEVICES=${device_groups[$slot]} \
    EVAL_TENSOR_PARALLEL_SIZE=4 \
    EVAL_BACKEND_PORT=$backend_port \
    EVAL_ROUTER_PORT=$router_port \
    EVAL_DATA_PARALLEL_RPC_PORT=$rpc_port \
    "$model_launcher" \
      "${models[$index]}" "${labels[$index]}" "${revisions[$index]}" &
  active_pids[$slot]=$!
  active_labels[$slot]=${labels[$index]}
}

next_pending=0
for ((slot = 0; slot < parallelism && next_pending < ${#pending[@]}; slot++)); do
  launch_slot "$slot" "${pending[$next_pending]}"
  next_pending=$((next_pending + 1))
done

while :; do
  active=0
  progressed=0
  for ((slot = 0; slot < parallelism; slot++)); do
    pid=${active_pids[$slot]:-}
    [[ -n "$pid" ]] || continue
    active=$((active + 1))
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi

    if ! wait "$pid"; then
      echo "qualification failed: ${active_labels[$slot]}" >&2
      exit 1
    fi
    active_pids[$slot]=
    active_labels[$slot]=
    active=$((active - 1))
    progressed=1

    if (( next_pending < ${#pending[@]} )); then
      launch_slot "$slot" "${pending[$next_pending]}"
      next_pending=$((next_pending + 1))
      active=$((active + 1))
    fi
  done

  (( active > 0 )) || break
  (( progressed > 0 )) || sleep 1
done

trap - EXIT INT TERM

echo "tranche qualification completed: $output_root"
