#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dataset=${BOOTSTRAP_DATASET:-/ephemeral/subagent-rung/data/281-qwen35-27b-prime-agent-teacher-bootstrap}
config=${TEACHER_CONFIG:-configs/debug/subagent-communication/283-qwen35-27b-prime-agent-teacher-bootstrap-online.toml}
base_revision=fc05daec18b0a78c049392ed2e771dde82bdf654

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if ! git diff-index --quiet HEAD -- || [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing to train from a dirty Prime-RL checkout" >&2
  exit 1
fi
if [[ -n "$(git submodule status --recursive | grep '^[+-]' || true)" ]]; then
  echo "refusing to train with mismatched submodule revisions" >&2
  exit 1
fi

manifest="$dataset/manifest.json"
train_data="$dataset/train.parquet"
if [[ ! -f "$manifest" || ! -f "$train_data" ]]; then
  echo "verified bootstrap dataset is incomplete: $dataset" >&2
  exit 1
fi

.venv/bin/python - "$manifest" "$base_revision" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
expected_revision = sys.argv[2]
if manifest.get("base_revision") != expected_revision:
    raise SystemExit(
        f"bootstrap base revision {manifest.get('base_revision')} != {expected_revision}"
    )
counts = manifest.get("counts", {})
requirements = {
    "reasoning.present_traces": 1,
    "instruction.standard.admitted_traces": 1,
    "instruction.guided.admitted_traces": 1,
}
ownership_families = (
    "json_sum",
    "csv_amount_total",
    "text_keyword_count",
    "markdown_heading_count",
    "log_error_count",
    "python_def_count",
    "json_max_value",
    "sha256_prefix",
)
for owner in ("child", "coordinator"):
    for family in ownership_families:
        requirements[f"ownership.{owner}.family.{family}"] = 1
for family in ("direct", "single", "parallel", "followup", "handshake"):
    requirements[f"family.{family}"] = 4
missing = [f"{key}={counts.get(key, 0)}<{minimum}" for key, minimum in requirements.items() if counts.get(key, 0) < minimum]
if missing:
    raise SystemExit(f"bootstrap manifest does not pass coverage: {', '.join(missing)}")
if manifest.get("rows", 0) <= 0:
    raise SystemExit("bootstrap manifest contains no training rows")
print(f"verified bootstrap manifest with {manifest['rows']} rows")
PY

if [[ ! -x .venv/bin/evaluator ]]; then
  echo "online evaluator is missing; run scripts/setup_prime_agent_mastery_host.sh" >&2
  exit 1
fi
if [[ ! -x .venv/bin/vllm-router ]]; then
  echo "vllm-router is missing; run scripts/setup_prime_agent_mastery_host.sh" >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import ownership_invariant_v1
import prime_agent_capabilities_v1
import subagent_communication_v1
PY

if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]] && [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

set -a
source .env
set +a
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN or HF_KEY is required" >&2
  exit 1
fi

exec .venv/bin/sft @ "$config"
