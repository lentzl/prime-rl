#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-r7-natural-yield-scaffold-y1-harvest}
evaluation_root=${PROCEDURAL_HARNESS_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-natural-yield-scaffold-v1}
start_index=${NATURAL_YIELD_SCAFFOLD_HARVEST_START_INDEX:-3510000}

NATURAL_YIELD_SCAFFOLD_ADMISSION_CONFIG=$experiment/natural-yield-scaffold-harvest.toml \
NATURAL_YIELD_SCAFFOLD_START_INDEX=$start_index \
PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  "$root/scripts/run_qwen35_27b_natural_yield_scaffold_admission_v1.sh" "$model" "$label"

summary=$evaluation_root/$label/train-admission/SCAFFOLD_SUMMARY.json
"$root/.venv/bin/python" - "$summary" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    report = json.load(handle)
if not report["admission"]["y1_harvest_ready"]:
    raise SystemExit("Y1 rejected: fewer than 16 verified scaffolded successes across 6 semantic families")
print("Y1_HARVEST_READY")
PY
