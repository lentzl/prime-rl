#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
template=${HARNESS_ACTION_TRAIN_CONFIG:-$experiment/harness-action-grpo.toml}
rung=${1:-atomic_send}
run_label=${2:-r1}
admission_summary=${HARNESS_ACTION_ADMISSION_SUMMARY:-/ephemeral/evals/qwen35-27b-procedural-harness-action-ramp-v1/untouched-$rung/train-admission/SUMMARY.json}
model_repo=${HARNESS_ACTION_MODEL_REPO:-Qwen/Qwen3.5-27B}
model_revision=${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}
train_start_index=${HARNESS_ACTION_TRAIN_START_INDEX:-1000000}
train_count=${HARNESS_ACTION_TRAIN_COUNT:-512}
train_lr=${HARNESS_ACTION_TRAIN_LR:-5e-7}
batch_size=${HARNESS_ACTION_BATCH_SIZE:-16}
max_steps=${HARNESS_ACTION_MAX_STEPS:-4}

case "$rung" in
  atomic_state|atomic_send|atomic_child_request|atomic_followup|atomic_parallel) ;;
  *) echo "unknown harness-action rung: $rung" >&2; exit 1 ;;
esac

cd "$root"
export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
if [[ ! -x .venv/bin/rl || ! -x .venv/bin/inference || ! -x .venv/bin/env-server ]]; then
  echo "Prime-RL training executables are missing" >&2
  exit 1
fi
if [[ ! -f "$template" ]]; then
  echo "harness-action training config does not exist: $template" >&2
  exit 1
fi
if [[ ! -f "$admission_summary" ]]; then
  echo "harness-action admission summary does not exist: $admission_summary" >&2
  exit 1
fi
if [[ ! "$train_start_index" =~ ^[0-9]+$ ]]; then
  echo "harness-action training start index must be non-negative: $train_start_index" >&2
  exit 1
fi
if [[ ! "$train_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "harness-action training count must be positive: $train_count" >&2
  exit 1
fi
if [[ ! "$max_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "harness-action max steps must be positive: $max_steps" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

"$root/.venv/bin/python" - "$admission_summary" "$rung" <<'PY'
import json
import sys

from scripts.summarize_procedural_harness_master_v1 import (
    select_curriculum_rung_admission,
)

with open(sys.argv[1]) as handle:
    report = json.load(handle)
select_curriculum_rung_admission(report, sys.argv[2])
PY

uv sync --frozen --inexact --extra flash-attn >/dev/null
if [[ ! -x .venv/bin/vllm-router ]]; then
  uv pip install --python "$root/.venv/bin/python" --no-deps \
    "vllm-router @ https://github.com/PrimeIntellect-ai/router/releases/download/v0.2.0/vllm_router-0.2.0-cp38-abi3-manylinux_2_28_x86_64.whl" \
    >/dev/null
fi
.venv/bin/python -c "import prime_rl.trainer.model"
for package in subagent_communication_v1 procedural_harness_master_v1; do
  uv pip install --python "$root/.venv/bin/python" --no-deps --editable \
    "$root/deps/verifiers/environments/$package" >/dev/null
done
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN or HF_KEY is required" >&2
  exit 1
fi

if [[ -n "${HARNESS_ACTION_MODEL_PATH:-}" ]]; then
  model_snapshot=$HARNESS_ACTION_MODEL_PATH
  if [[ ! -d "$model_snapshot" ]]; then
    echo "HARNESS_ACTION_MODEL_PATH is not a directory: $model_snapshot" >&2
    exit 1
  fi
else
  model_snapshot=$(
    .venv/bin/python - "$model_repo" "$model_revision" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
PY
  )
fi

resolved_config=$(mktemp --suffix=.toml)
trap 'rm -f "$resolved_config"' EXIT
.venv/bin/python - "$template" "$resolved_config" "$rung" "$run_label" "$train_start_index" "$train_count" "$train_lr" "$batch_size" "$max_steps" <<'PY'
import math
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
run_name = f"{sys.argv[3].replace('_', '-')}-grpo-{sys.argv[4]}"
learning_rate = float(sys.argv[7])
batch_size = int(sys.argv[8])
max_steps = int(sys.argv[9])
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise SystemExit(f"training learning rate must be positive and finite: {sys.argv[7]}")
if batch_size <= 0 or batch_size % 8:
    raise SystemExit(f"training batch size must be a positive multiple of group size 8: {batch_size}")
oversampling_factor = 8 / batch_size
patterns = (
    (r"^max_steps = [0-9]+$", f"max_steps = {max_steps}"),
    (r'^curriculum_rung = "[^"]+"$', f'curriculum_rung = "{sys.argv[3]}"'),
    (r'^name = "atomic-state-grpo"$', f'name = "{run_name}"'),
    (r'^dir = "atomic-state-grpo"$', f'dir = "{run_name}"'),
    (r'^start_index = [0-9]+$', f'start_index = {sys.argv[5]}'),
    (r'^count = [0-9]+$', f'count = {sys.argv[6]}'),
    (r'^lr = [^\n]+$', f'lr = {learning_rate}'),
    (r'^batch_size = [0-9]+$', f'batch_size = {batch_size}'),
    (r'^oversampling_factor = [^\n]+$', f'oversampling_factor = {oversampling_factor}'),
)
for pattern, replacement in patterns:
    source, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"training config did not match {pattern}")
Path(sys.argv[2]).write_text(source)
PY
config=$resolved_config

if [[ "${HARNESS_ACTION_TRAIN_DRY_RUN:-false}" == true ]]; then
  rl @ "$config" --model.name "$model_snapshot" --dry-run
  echo "harness-action $rung hard-GRPO preflight passed"
  exit 0
fi

rl @ "$config" --model.name "$model_snapshot"
