#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
eval_experiment=experiments/qwen35-27b-procedural-harness-master-v1
train_run=${1:-/ephemeral/outputs/qwen35-27b-procedural-harness-master-v1/bootstrap-grpo}
label=${2:-checkpoint-battery-r1}
evaluation_root=${PROCEDURAL_HARNESS_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-procedural-harness-master-v1}/$label
server_driver=$root/scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh
eval_driver=$root/scripts/run_qwen35_27b_procedural_harness_master_baseline_v1.sh
model_revision=${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}

cd "$root"
export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
if [[ ! -x "$server_driver" || ! -x "$eval_driver" ]]; then
  echo "procedural checkpoint evaluation drivers are not executable" >&2
  exit 1
fi
if [[ -e "$evaluation_root" ]]; then
  echo "refusing to overwrite checkpoint battery: $evaluation_root" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to evaluate while another GPU process is active" >&2
  exit 1
fi

max_steps=$(
  .venv/bin/python - "$experiment/bootstrap-grpo.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    print(tomllib.load(handle)["max_steps"])
PY
)
if [[ ! "$max_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "training config has an invalid max_steps: $max_steps" >&2
  exit 1
fi

models=()
labels=()
model_snapshot=$(
  .venv/bin/python - "$model_revision" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download("Qwen/Qwen3.5-27B", revision=sys.argv[1]))
PY
)
if [[ "$(basename "$model_snapshot")" != "$model_revision" ]]; then
  echo "resolved untouched snapshot does not match the pinned revision" >&2
  exit 1
fi
models+=("$model_snapshot")
labels+=("untouched")

for step in $(seq 1 "$max_steps"); do
  weights=$train_run/weights/step_$step
  if [[ ! -f "$weights/STABLE" ]]; then
    echo "checkpoint step $step is absent or unstable: $weights" >&2
    exit 1
  fi
  if [[ ! -f "$weights/model.safetensors.index.json" ]]; then
    echo "checkpoint step $step lacks its sharded safetensors index: $weights" >&2
    exit 1
  fi
  models+=("$weights")
  labels+=("step-$step")
done

mkdir -p "$evaluation_root"
{
  printf 'prime_rl_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'verifiers_commit=%s\n' "$(git -C deps/verifiers rev-parse HEAD)"
  printf 'model_revision=%s\n' "$model_revision"
  printf 'training_run=%s\n' "$train_run"
  printf 'max_steps=%s\n' "$max_steps"
} >"$evaluation_root/VERSIONS.txt"

for index in "${!models[@]}"; do
  model=${models[$index]}
  candidate=${labels[$index]}
  PRIME_MASTERY_OUTPUT_ROOT="$evaluation_root" \
    EVAL_DRIVER="$eval_driver" \
    EVAL_EXPERIMENT_DIR="$eval_experiment" \
    EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-4}" \
    "$server_driver" "$model" "$candidate" "$model_revision"
done

.venv/bin/python scripts/compare_procedural_harness_master_checkpoints_v1.py \
  "$evaluation_root" --expected-steps "$max_steps" \
  --output "$evaluation_root/MATRIX.json"
echo "procedural checkpoint battery completed: $evaluation_root"
