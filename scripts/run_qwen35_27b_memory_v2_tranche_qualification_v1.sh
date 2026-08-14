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
cleanup() {
  trap - EXIT INT TERM
  for pid in "${active_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${active_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for ((offset = 0; offset < ${#pending[@]}; offset += parallelism)); do
  active_pids=()
  active_labels=()
  for ((slot = 0; slot < parallelism && offset + slot < ${#pending[@]}; slot++)); do
    index=${pending[$((offset + slot))]}
    backend_port=$((8100 + 100 * slot))
    router_port=$((8000 + 100 * slot))
    rpc_port=$((13345 + 100 * slot))
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
    active_pids+=("$!")
    active_labels+=("${labels[$index]}")
  done

  failed=0
  for index in "${!active_pids[@]}"; do
    if ! wait "${active_pids[$index]}"; then
      echo "qualification failed: ${active_labels[$index]}" >&2
      failed=1
    fi
  done
  active_pids=()
  if (( failed )); then
    exit 1
  fi
done

trap - EXIT INT TERM

echo "tranche qualification completed: $output_root"
