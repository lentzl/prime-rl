#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
admission_summary=${PROCEDURAL_HARNESS_ADMISSION_SUMMARY:-/ephemeral/evals/qwen35-27b-procedural-harness-master-v1/untouched-admission-v2-r1/train-admission/SUMMARY.json}
model_revision=${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}

cd "$root"
export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
if [[ ! -x .venv/bin/rl || ! -x .venv/bin/inference || ! -x .venv/bin/env-server ]]; then
  echo "Prime-RL training executables are missing" >&2
  exit 1
fi
if [[ ! -f "$admission_summary" ]]; then
  echo "procedural admission summary does not exist: $admission_summary" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

# The base lock keeps FlashAttention optional; hydrate its pinned wheel without
# removing editable task environments from an already prepared machine.
uv sync --frozen --inexact --extra flash-attn >/dev/null
if [[ ! -x .venv/bin/vllm-router ]]; then
  uv pip install --python "$root/.venv/bin/python" --no-deps \
    "vllm-router @ https://github.com/PrimeIntellect-ai/router/releases/download/v0.2.0/vllm_router-0.2.0-cp38-abi3-manylinux_2_28_x86_64.whl" \
    >/dev/null
fi
.venv/bin/python -c "import prime_rl.trainer.model"

selection=$(
  .venv/bin/python - "$admission_summary" <<'PY'
import json
import sys

from scripts.summarize_procedural_harness_master_v1 import select_training_mode

with open(sys.argv[1]) as handle:
    report = json.load(handle)
try:
    mode, families = select_training_mode(report)
except ValueError as error:
    raise SystemExit(str(error)) from error
print(mode + "|" + ",".join(families))
PY
)
mode=${selection%%|*}
admitted_families=${selection#*|}
if [[ "$mode" == hard ]]; then
  default_config=$experiment/bootstrap-grpo.toml
else
  default_config=$experiment/bootstrap-shaped-grpo.toml
fi
config=${PROCEDURAL_HARNESS_TRAIN_CONFIG:-$default_config}
if [[ ! -f "$config" ]]; then
  echo "procedural training config does not exist: $config" >&2
  exit 1
fi
.venv/bin/python - "$config" "$mode" <<'PY'
import sys
from pathlib import Path

from scripts.summarize_procedural_harness_master_v1 import configured_reward_mode

actual = configured_reward_mode(Path(sys.argv[1]))
expected = "hard" if sys.argv[2] == "hard" else "bootstrap"
if actual != expected:
    raise SystemExit(f"selected {expected} reward but config uses {actual}")
PY
printf 'selected %s reward from admission; informative families: %s\n' "$mode" "$admitted_families"

resolved_config=$(mktemp --suffix=.toml)
trap 'rm -f "$resolved_config"' EXIT
.venv/bin/python - "$config" "$resolved_config" "$admitted_families" <<'PY'
import json
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
families = sys.argv[3].split(",")
rendered, replacements = re.subn(
    r"^families = \[.*\]$",
    "families = " + json.dumps(families),
    source,
    count=1,
    flags=re.MULTILINE,
)
if replacements != 1:
    raise SystemExit("procedural training config must contain one families list")
Path(sys.argv[2]).write_text(rendered)
PY
config=$resolved_config

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

model_snapshot=$(
  .venv/bin/python - "$model_revision" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download("Qwen/Qwen3.5-27B", revision=sys.argv[1]))
PY
)
if [[ "$(basename "$model_snapshot")" != "$model_revision" ]]; then
  echo "resolved model snapshot does not match the pinned revision" >&2
  exit 1
fi

if [[ "${PROCEDURAL_HARNESS_TRAIN_DRY_RUN:-false}" == true ]]; then
  rl @ "$config" --model.name "$model_snapshot" --dry-run
  echo "procedural Harness Master bootstrap preflight passed"
  exit 0
fi

rl @ "$config" --model.name "$model_snapshot"
