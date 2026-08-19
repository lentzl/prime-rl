#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${NATURAL_YIELD_SDPO_CUMULATIVE_CONFIG:-$root/experiments/qwen35-27b-procedural-harness-master-v1/natural-yield-sdpo-cumulative.toml}
run_dir=/ephemeral/outputs/qwen35-27b-natural-yield-sdpo-v1/cumulative-r1
prerequisite=${NATURAL_YIELD_SDPO_AUDIT:-/ephemeral/outputs/qwen35-27b-natural-yield-sdpo-v1/zero-lr-audit/AUDIT.json}
local_r7=${HARNESS_ACTION_MODEL_PATH:-/ephemeral/outputs/qwen35-27b-procedural-harness-action-ramp-v1/atomic-send-grpo-r7/weights/step_1}
model_repo=${HARNESS_ACTION_MODEL_REPO:-lentzl/rlm-prime-agent-qwen35-27b-harness-r7-20260818}
model_revision=${MODEL_REVISION:-8f0568faed72d0db2e2258c18b1aabdcefd680cc}
validator_module=scripts.validate_natural_yield_sdpo_cumulative_v1

cd "$root"
export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

for path in "$config" "$prerequisite"; do
  if [[ ! -f "$path" ]]; then
    echo "required cumulative natural-yield input is absent: $path" >&2
    exit 1
  fi
done
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite cumulative natural-yield update: $run_dir" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

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

if [[ -d "$local_r7" && -f "$local_r7/STABLE" ]]; then
  model_snapshot=$local_r7
else
  if [[ -z "$HF_TOKEN" ]]; then
    echo "canonical R7 is not local and HF_TOKEN or HF_KEY is unavailable" >&2
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

.venv/bin/python - "$prerequisite" "$model_snapshot" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    audit = json.load(handle)
if (
    audit.get("verdict") != "pass"
    or audit.get("mechanism") != "natural-yield-feedback-conditioned-sdpo-zero-lr"
    or audit.get("expected_model_path") != sys.argv[2]
    or audit.get("model_artifacts_written") is not False
):
    raise SystemExit("natural-yield zero-LR audit does not authorize an update")
PY

if [[ "${NATURAL_YIELD_SDPO_CUMULATIVE_DRY_RUN:-false}" == true ]]; then
  dry_run_root=$(mktemp -d /tmp/natural-yield-sdpo-cumulative-dry-run.XXXXXX)
  trap 'rm -rf "$dry_run_root"' EXIT
  rl @ "$config" \
    --model.name "$model_snapshot" \
    --output-dir "$dry_run_root" \
    --run.dir preflight \
    --dry-run
  echo "cumulative natural-yield SDPO preflight passed"
  exit 0
fi

rl @ "$config" --model.name "$model_snapshot"

for step in 1 2 3 4; do
  .venv/bin/python scripts/finalize_hf_processor_metadata.py \
    "$model_snapshot" "$run_dir/weights/step_$step"
done
.venv/bin/python -m "$validator_module" \
  "$run_dir" \
  --prerequisite "$prerequisite" \
  --expected-model-path "$model_snapshot" \
  --output "$run_dir/CUMULATIVE_UPDATE.json"
