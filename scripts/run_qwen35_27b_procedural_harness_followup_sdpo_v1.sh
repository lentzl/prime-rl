#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
template=${HARNESS_FOLLOWUP_SDPO_CONFIG:-$experiment/harness-followup-sdpo.toml}
run_label=${1:-r1}
admission_summary=${HARNESS_FOLLOWUP_ADMISSION_SUMMARY:-/ephemeral/evals/qwen35-27b-procedural-harness-action-ramp-v1/followup-feedback-admission-r1/train-admission/SUMMARY.json}
feedback_audit=${HARNESS_FOLLOWUP_FEEDBACK_AUDIT:-/ephemeral/evals/qwen35-27b-procedural-harness-action-ramp-v1/followup-feedback-admission-r1/train-admission/FEEDBACK_AUDIT.json}
model_repo=${HARNESS_ACTION_MODEL_REPO:-Qwen/Qwen3.5-27B}
model_revision=${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}
train_start_index=${HARNESS_FOLLOWUP_TRAIN_START_INDEX:-1100000}
train_count=${HARNESS_FOLLOWUP_TRAIN_COUNT:-512}
train_lr=${HARNESS_FOLLOWUP_TRAIN_LR:-2.5e-7}
batch_size=${HARNESS_FOLLOWUP_BATCH_SIZE:-16}

cd "$root"
export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

for path in "$template" "$admission_summary" "$feedback_audit"; do
  if [[ ! -f "$path" ]]; then
    echo "required follow-up SDPO input does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! "$train_start_index" =~ ^[0-9]+$ ]]; then
  echo "follow-up SDPO start index must be non-negative: $train_start_index" >&2
  exit 1
fi
if [[ ! "$train_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "follow-up SDPO count must be positive: $train_count" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

"$root/.venv/bin/python" - "$admission_summary" "$feedback_audit" <<'PY'
import json
import sys

from scripts.summarize_procedural_harness_master_v1 import (
    classify_curriculum_rung_admission,
)

with open(sys.argv[1]) as handle:
    admission = json.load(handle)
if classify_curriculum_rung_admission(admission, "atomic_followup") != "disconnected":
    raise SystemExit("follow-up SDPO requires a reward-disconnected hard admission")
with open(sys.argv[2]) as handle:
    audit = json.load(handle)
if (
    audit.get("schema_version") != "prime-agent/procedural-followup-feedback/v1"
    or audit.get("episodes", 0) < 8
    or audit.get("episodes", 0) % 8 != 0
    or audit.get("active_feedback_traces", 0) < 2
    or audit.get("structural_routing_verified") is not True
    or audit.get("routing_contract") != "one-mask-per-trainable-branch"
    or audit.get("answer_free") is not True
    or audit.get("failure_local") is not True
):
    raise SystemExit("follow-up feedback audit does not authorize SDPO")
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

if [[ -n "${HARNESS_ACTION_MODEL_PATH:-}" ]]; then
  model_snapshot=$HARNESS_ACTION_MODEL_PATH
  if [[ ! -d "$model_snapshot" || ! -f "$model_snapshot/STABLE" ]]; then
    echo "HARNESS_ACTION_MODEL_PATH is not a stable checkpoint: $model_snapshot" >&2
    exit 1
  fi
else
  if [[ -z "$HF_TOKEN" ]]; then
    echo "HF_TOKEN or HF_KEY is required when no local model path is supplied" >&2
    exit 1
  fi
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
.venv/bin/python - "$template" "$resolved_config" "$run_label" "$train_start_index" "$train_count" "$train_lr" "$batch_size" <<'PY'
import math
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
run_name = f"atomic-followup-sdpo-{sys.argv[3]}"
learning_rate = float(sys.argv[6])
batch_size = int(sys.argv[7])
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise SystemExit(f"follow-up SDPO learning rate must be positive and finite: {sys.argv[6]}")
if batch_size <= 0:
    raise SystemExit(f"follow-up SDPO batch size must be positive: {batch_size}")
oversampling_factor = 8 / batch_size
patterns = (
    (r'^name = "atomic-followup-sdpo"$', f'name = "{run_name}"'),
    (r'^dir = "atomic-followup-sdpo"$', f'dir = "{run_name}"'),
    (r'^start_index = [0-9]+$', f'start_index = {sys.argv[4]}'),
    (r'^count = [0-9]+$', f'count = {sys.argv[5]}'),
    (r'^lr = [^\n]+$', f'lr = {learning_rate}'),
    (r'^batch_size = [0-9]+$', f'batch_size = {batch_size}'),
    (r'^oversampling_factor = [^\n]+$', f'oversampling_factor = {oversampling_factor}'),
)
for pattern, replacement in patterns:
    source, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"follow-up SDPO config did not match {pattern}")
Path(sys.argv[2]).write_text(source)
PY

if [[ "${HARNESS_FOLLOWUP_SDPO_DRY_RUN:-false}" == true ]]; then
  rl @ "$resolved_config" --model.name "$model_snapshot" --dry-run
  echo "failure-local follow-up SDPO preflight passed"
  exit 0
fi

rl @ "$resolved_config" --model.name "$model_snapshot"
