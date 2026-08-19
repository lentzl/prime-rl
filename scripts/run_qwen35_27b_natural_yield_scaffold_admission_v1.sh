#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
template=${NATURAL_YIELD_SCAFFOLD_ADMISSION_CONFIG:-$experiment/natural-yield-scaffold-admission.toml}
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-r7-natural-yield-scaffold-y0}
evaluation_root=${PROCEDURAL_HARNESS_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-natural-yield-scaffold-v1}
start_index=${NATURAL_YIELD_SCAFFOLD_START_INDEX:-3500000}

if [[ ! -f "$template" ]]; then
  echo "scaffold admission config does not exist: $template" >&2
  exit 1
fi
if [[ ! "$start_index" =~ ^[0-9]+$ ]]; then
  echo "scaffold admission start index must be non-negative: $start_index" >&2
  exit 1
fi
if [[ ! -f "$root/deps/verifiers/environments/procedural_harness_master_v1/procedural_harness_master_v1/natural_yield_scaffold.py" ]]; then
  echo "Verifiers constrained-yield scaffold is not present in the pinned submodule" >&2
  exit 1
fi

resolved_config=$(mktemp --suffix=.toml)
trap 'rm -f "$resolved_config"' EXIT
"$root/.venv/bin/python" - "$template" "$resolved_config" "$start_index" <<'PY'
import re
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text()
source, count = re.subn(r"^start_index = [0-9]+$", f"start_index = {sys.argv[3]}", source, count=1, flags=re.MULTILINE)
if count != 1:
    raise SystemExit("scaffold admission config must contain one start_index")
Path(sys.argv[2]).write_text(source)
PY

# Training-only request mediation. The frozen task/scorer and model-visible prompt are unchanged.
export PROCEDURAL_NATURAL_YIELD_SCAFFOLD=1
NATURAL_POLICY_ADMISSION_CONFIG=$resolved_config \
NATURAL_POLICY_RUNG=natural_n1 \
NATURAL_POLICY_ADMISSION_START_INDEX=$start_index \
PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  "$root/scripts/run_qwen35_27b_natural_policy_admission_v1.sh" "$model" "$label" || true

run_dir=$evaluation_root/$label/train-admission
if [[ ! -f "$run_dir/traces.jsonl" ]]; then
  echo "missing scaffold admission traces: $run_dir/traces.jsonl" >&2
  exit 1
fi
"$root/.venv/bin/python" -m scripts.summarize_natural_yield_scaffold_v1 "$run_dir" > "$run_dir/SCAFFOLD_SUMMARY.json"
cat "$run_dir/SCAFFOLD_SUMMARY.json"

"$root/.venv/bin/python" - "$run_dir/SCAFFOLD_SUMMARY.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    report = json.load(handle)
if report["errors"] != 0:
    raise SystemExit("Y0 rejected: rollout errors present")
if not report["admission"]["y0_connected"]:
    raise SystemExit("Y0 rejected: constrained exploration did not create enough verified native successes")
print("Y0_CONNECTED")
PY
