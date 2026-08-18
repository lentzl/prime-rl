#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
template=${HARNESS_SUCCESS_SFT_CONFIG:-$experiment/harness-success-sft.toml}
rung=${1:-atomic_child_request}
run_label=${2:-r9}
admission_summary=${HARNESS_SUCCESS_ADMISSION_SUMMARY:-/ephemeral/evals/qwen35-27b-procedural-harness-action-ramp-v1/r7-paired-r8retry1-2300000-atomic_child_request-gate-r1/train-admission/SUMMARY.json}
model_repo=${HARNESS_SUCCESS_MODEL_REPO:-lentzl/rlm-prime-agent-qwen35-27b-harness-r7-20260818}
model_revision=${HARNESS_SUCCESS_MODEL_REVISION:-8f0568faed72d0db2e2258c18b1aabdcefd680cc}
train_start_index=${HARNESS_SUCCESS_TRAIN_START_INDEX:-2400000}
train_count=${HARNESS_SUCCESS_TRAIN_COUNT:-512}
train_lr=${HARNESS_SUCCESS_TRAIN_LR:-1e-7}
batch_size=${HARNESS_SUCCESS_BATCH_SIZE:-16}

case "$rung" in
  atomic_send|atomic_child_request) ;;
  *) echo "unsupported success-SFT rung: $rung" >&2; exit 1 ;;
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
  echo "success-SFT config does not exist: $template" >&2
  exit 1
fi
if [[ ! -f "$admission_summary" ]]; then
  echo "success-SFT admission summary does not exist: $admission_summary" >&2
  exit 1
fi
if [[ ! "$train_start_index" =~ ^[0-9]+$ ]]; then
  echo "success-SFT start index must be non-negative: $train_start_index" >&2
  exit 1
fi
if [[ ! "$train_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "success-SFT training count must be positive: $train_count" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

"$root/.venv/bin/python" - "$admission_summary" "$rung" <<'PY'
import json
import sys

from scripts.summarize_procedural_harness_master_v1 import select_curriculum_rung_admission

with open(sys.argv[1]) as handle:
    report = json.load(handle)
select_curriculum_rung_admission(report, sys.argv[2])
PY

"$root/scripts/build_prime_agent_runtime_image_v1.sh" >/dev/null
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

if [[ -n "${HARNESS_SUCCESS_MODEL_PATH:-}" ]]; then
  model_snapshot=$HARNESS_SUCCESS_MODEL_PATH
else
  model_snapshot=$(
    .venv/bin/python - "$model_repo" "$model_revision" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
PY
  )
fi
if [[ ! -f "$model_snapshot/STABLE" || ! -f "$model_snapshot/model.safetensors.index.json" ]]; then
  echo "HARNESS_SUCCESS_MODEL_PATH is not a stable full-weight checkpoint: $model_snapshot" >&2
  exit 1
fi

resolved_config=$(mktemp --suffix=.toml)
trap 'rm -f "$resolved_config"' EXIT
.venv/bin/python - "$template" "$resolved_config" "$rung" "$run_label" "$train_start_index" "$train_count" "$train_lr" "$batch_size" "$model_snapshot" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
run_name = f"{sys.argv[3].replace('_', '-')}-success-sft-{sys.argv[4]}"
learning_rate = float(sys.argv[7])
batch_size = int(sys.argv[8])
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise SystemExit(f"success-SFT learning rate must be positive and finite: {sys.argv[7]}")
if batch_size <= 0:
    raise SystemExit(f"success-SFT batch size must be positive: {batch_size}")
oversampling_factor = 8 / batch_size
patterns = (
    (r'^curriculum_rung = "[^"]+"$', f'curriculum_rung = "{sys.argv[3]}"'),
    (r'^name = "atomic-child-request-success-sft-r9"$', f'name = "{run_name}"'),
    (r'^dir = "atomic-child-request-success-sft-r9"$', f'dir = "{run_name}"'),
    (r'^start_index = [0-9]+$', f'start_index = {sys.argv[5]}'),
    (r'^count = [0-9]+$', f'count = {sys.argv[6]}'),
    (r'^lr = [^\n]+$', f'lr = {learning_rate}'),
    (r'^batch_size = [0-9]+$', f'batch_size = {batch_size}'),
    (r'^oversampling_factor = [^\n]+$', f'oversampling_factor = {oversampling_factor}'),
    (r'^name = "__MANAGED_R7_ENDPOINT__"$', f'name = {json.dumps(sys.argv[9])}'),
)
for pattern, replacement in patterns:
    source, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"success-SFT config did not match {pattern}")
Path(sys.argv[2]).write_text(source)
PY

if [[ "${HARNESS_SUCCESS_SFT_DRY_RUN:-false}" == true ]]; then
  rl @ "$resolved_config" --model.name "$model_snapshot" --dry-run
  echo "harness success-SFT preflight passed"
  exit 0
fi

rl @ "$resolved_config" --model.name "$model_snapshot"
