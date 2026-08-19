#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
template=${NATURAL_YIELD_SCAFFOLD_SFT_CONFIG:-$experiment/natural-yield-scaffold-sft.toml}
run_label=${1:-y2-routing-audit}
harvest_summary=${NATURAL_YIELD_SCAFFOLD_HARVEST_SUMMARY:-/ephemeral/evals/qwen35-27b-natural-yield-scaffold-v1/r7-natural-yield-scaffold-y1-harvest/train-admission/SCAFFOLD_SUMMARY.json}
model_repo=${NATURAL_YIELD_SCAFFOLD_MODEL_REPO:-lentzl/rlm-prime-agent-qwen35-27b-harness-r7-20260818}
model_revision=${NATURAL_YIELD_SCAFFOLD_MODEL_REVISION:-8f0568faed72d0db2e2258c18b1aabdcefd680cc}
train_start_index=${NATURAL_YIELD_SCAFFOLD_TRAIN_START_INDEX:-3520000}
train_lr=${NATURAL_YIELD_SCAFFOLD_TRAIN_LR:-0}

cd "$root"
export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
if [[ ! -f "$template" || ! -f "$harvest_summary" ]]; then
  echo "missing scaffold SFT config or Y1 harvest summary" >&2
  exit 1
fi
if [[ ! -f "$root/deps/verifiers/environments/procedural_harness_master_v1/procedural_harness_master_v1/natural_yield_scaffold.py" ]]; then
  echo "Verifiers constrained-yield scaffold is not present in the pinned submodule" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

.venv/bin/python - "$harvest_summary" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    report = json.load(handle)
if not report["admission"]["y1_harvest_ready"]:
    raise SystemExit("Y2 blocked: Y1 harvest is not diversity-ready")
PY

"$root/scripts/build_prime_agent_runtime_image_v1.sh" >/dev/null
uv sync --frozen --inexact --extra flash-attn >/dev/null
for package in subagent_communication_v1 procedural_harness_master_v1; do
  uv pip install --python "$root/.venv/bin/python" --no-deps --editable \
    "$root/deps/verifiers/environments/$package" >/dev/null
done
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
if [[ -z "$HF_TOKEN" ]]; then echo "HF_TOKEN or HF_KEY is required" >&2; exit 1; fi

if [[ -n "${NATURAL_YIELD_SCAFFOLD_MODEL_PATH:-}" ]]; then
  model_snapshot=$NATURAL_YIELD_SCAFFOLD_MODEL_PATH
else
  model_snapshot=$(.venv/bin/python - "$model_repo" "$model_revision" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
PY
)
fi
if [[ ! -f "$model_snapshot/STABLE" || ! -f "$model_snapshot/model.safetensors.index.json" ]]; then
  echo "source is not a stable full-weight R7 checkpoint: $model_snapshot" >&2
  exit 1
fi

resolved=$(mktemp --suffix=.toml)
trap 'rm -f "$resolved"' EXIT
.venv/bin/python - "$template" "$resolved" "$run_label" "$train_start_index" "$train_lr" "$model_snapshot" <<'PY'
import json, math, re, sys
from pathlib import Path
source = Path(sys.argv[1]).read_text()
lr = float(sys.argv[5])
if not math.isfinite(lr) or lr < 0:
    raise SystemExit("learning rate must be finite and non-negative")
run_name = f"natural-yield-scaffold-sft-{sys.argv[3]}"
for pattern, replacement in (
    (r'^name = "natural-yield-scaffold-sft-y2"$', f'name = "{run_name}"'),
    (r'^dir = "natural-yield-scaffold-sft-y2"$', f'dir = "{run_name}"'),
    (r'^start_index = [0-9]+$', f'start_index = {sys.argv[4]}'),
    (r'^lr = [^\n]+$', f'lr = {lr}'),
    (r'^name = "__MANAGED_R7_ENDPOINT__"$', f'name = {json.dumps(sys.argv[6])}'),
):
    source, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"scaffold SFT config did not match {pattern}")
Path(sys.argv[2]).write_text(source)
PY

# This flag is deliberately collection-only. Never export it in Y3/frozen evaluation.
export PROCEDURAL_NATURAL_YIELD_SCAFFOLD=1
if [[ "${NATURAL_YIELD_SCAFFOLD_DRY_RUN:-false}" == true ]]; then
  rl @ "$resolved" --model.name "$model_snapshot" --dry-run
  exit 0
fi
rl @ "$resolved" --model.name "$model_snapshot"

if [[ "$train_lr" == "0" || "$train_lr" == "0.0" ]]; then
  echo "Y2_ROUTING_AUDIT_COMPLETE_NO_WEIGHT_CHANGE"
  echo "Inspect token export: every nonzero CE token must belong to scaffolded_yield_node_index; authorize nonzero Y2 only after that audit passes."
else
  echo "Y2_CANDIDATE_WRITTEN_UNPROMOTED"
  echo "Unset PROCEDURAL_NATURAL_YIELD_SCAFFOLD before every Y3 gate."
fi
