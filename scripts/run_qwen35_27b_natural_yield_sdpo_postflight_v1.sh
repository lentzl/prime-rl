#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base_model=${1:?canonical R7 path is required}
candidate_model=${2:?candidate path is required}
evaluation_root=${NATURAL_YIELD_POSTFLIGHT_ROOT:-/ephemeral/evals/qwen35-27b-natural-yield-sdpo-v1/low-dose-r1-gates}
server_driver=$root/scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh
gate_driver=$root/scripts/run_qwen35_27b_natural_yield_sdpo_gate_battery_v1.sh
model_revision=${MODEL_REVISION:-8f0568faed72d0db2e2258c18b1aabdcefd680cc}

cd "$root"
for model in "$base_model" "$candidate_model"; do
  if [[ ! -f "$model/STABLE" ]]; then
    echo "postflight model is not a stable checkpoint: $model" >&2
    exit 1
  fi
done
if [[ -e "$evaluation_root" ]]; then
  echo "refusing to overwrite natural-yield postflight: $evaluation_root" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to evaluate while another GPU process is active" >&2
  exit 1
fi

mkdir -p "$evaluation_root"
{
  printf 'prime_rl_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'verifiers_commit=%s\n' "$(git -C deps/verifiers rev-parse HEAD)"
  printf 'base_model=%s\n' "$base_model"
  printf 'candidate_model=%s\n' "$candidate_model"
  printf 'model_revision=%s\n' "$model_revision"
} >"$evaluation_root/VERSIONS.txt"

for item in "r7:$base_model" "candidate:$candidate_model"; do
  label=${item%%:*}
  model=${item#*:}
  PRIME_MASTERY_OUTPUT_ROOT=$evaluation_root \
  PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  EVAL_DRIVER=$gate_driver \
  EVAL_EXPERIMENT_DIR=experiments/qwen35-27b-procedural-harness-master-v1 \
  EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3} \
  EVAL_TENSOR_PARALLEL_SIZE=${EVAL_TENSOR_PARALLEL_SIZE:-4} \
    "$server_driver" "$model" "$label" "$model_revision"
done

.venv/bin/python -m scripts.compare_natural_yield_sdpo_gates_v1 \
  "$evaluation_root" r7 candidate \
  --output "$evaluation_root/COMPARISON.json"
echo "natural-yield SDPO postflight completed: $evaluation_root"
