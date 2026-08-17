#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
template=${HARNESS_CUMULATIVE_CONFIG:-$experiment/harness-send-followup-cumulative-grpo.toml}
run_label=${1:-r1}
send_admission=${HARNESS_CUMULATIVE_SEND_ADMISSION_SUMMARY:-/ephemeral/evals/qwen35-27b-procedural-harness-action-ramp-v1/step6-atomic-send-r1/train-admission/SUMMARY.json}
followup_admission=${HARNESS_CUMULATIVE_FOLLOWUP_ADMISSION_SUMMARY:-/ephemeral/evals/qwen35-27b-procedural-harness-action-ramp-v1/followup-feedback-admission-r5/train-admission/SUMMARY.json}
model_repo=${HARNESS_ACTION_MODEL_REPO:-Qwen/Qwen3.5-27B}
model_revision=${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}
send_start_index=${HARNESS_CUMULATIVE_SEND_START_INDEX:-1300000}
followup_start_index=${HARNESS_CUMULATIVE_FOLLOWUP_START_INDEX:-1400000}
train_lr=${HARNESS_CUMULATIVE_TRAIN_LR:-1.25e-7}
batch_size=${HARNESS_CUMULATIVE_BATCH_SIZE:-32}

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
  echo "cumulative harness-action training config does not exist: $template" >&2
  exit 1
fi
for summary in "$send_admission" "$followup_admission"; do
  if [[ ! -f "$summary" ]]; then
    echo "cumulative harness-action admission summary does not exist: $summary" >&2
    exit 1
  fi
done
for value in "$send_start_index" "$followup_start_index"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "cumulative training start index must be non-negative: $value" >&2
    exit 1
  fi
done
if [[ "$send_start_index" == "$followup_start_index" ]]; then
  echo "send and follow-up training windows must be disjoint" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

"$root/.venv/bin/python" - "$send_admission" "$followup_admission" <<'PY'
import json
import sys

from scripts.summarize_procedural_harness_master_v1 import select_curriculum_rung_admission

for path, rung in zip(sys.argv[1:], ("atomic_send", "atomic_followup"), strict=True):
    with open(path) as handle:
        report = json.load(handle)
    select_curriculum_rung_admission(report, rung)
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
  if [[ ! -d "$model_snapshot" || ! -f "$model_snapshot/STABLE" ]]; then
    echo "HARNESS_ACTION_MODEL_PATH is not a stable checkpoint: $model_snapshot" >&2
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
.venv/bin/python - "$template" "$resolved_config" "$run_label" "$send_start_index" "$followup_start_index" "$train_lr" "$batch_size" <<'PY'
import math
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
run_name = f"send-followup-cumulative-grpo-{sys.argv[3]}"
learning_rate = float(sys.argv[6])
batch_size = int(sys.argv[7])
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise SystemExit(f"cumulative training learning rate must be positive and finite: {sys.argv[6]}")
if batch_size <= 0 or batch_size % 8:
    raise SystemExit(f"cumulative training batch size must be a positive multiple of group size 8: {batch_size}")

replacements = (
    ('name = "send-followup-cumulative-grpo"', f'name = "{run_name}"'),
    ('dir = "send-followup-cumulative-grpo"', f'dir = "{run_name}"'),
    ('lr = 1.25e-7', f'lr = {learning_rate}'),
    ('batch_size = 32', f'batch_size = {batch_size}'),
    ('oversampling_factor = 0.25', f'oversampling_factor = {8 / batch_size}'),
    ('start_index = 1300000', f'start_index = {sys.argv[4]}'),
    ('start_index = 1400000', f'start_index = {sys.argv[5]}'),
)
for expected, replacement in replacements:
    source, count = re.subn(f"^{re.escape(expected)}$", replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"cumulative training config did not contain exactly one {expected!r}")
Path(sys.argv[2]).write_text(source)
PY

if [[ "${HARNESS_CUMULATIVE_TRAIN_DRY_RUN:-false}" == true ]]; then
  rl @ "$resolved_config" --model.name "$model_snapshot" --dry-run
  echo "send/follow-up cumulative hard-GRPO preflight passed"
  exit 0
fi

rl @ "$resolved_config" --model.name "$model_snapshot"
