#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base_model=${1:?R7 model path is required}
candidate_model=${2:?candidate model path is required}
base_label=${3:-r7-y3}
candidate_label=${4:-candidate-y3}
evaluation_root=${PROCEDURAL_HARNESS_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-natural-yield-scaffold-v1/y3-gates}

# Critical: Y3 is a native-policy test. Any scaffold leakage invalidates the result.
unset PROCEDURAL_NATURAL_YIELD_SCAFFOLD || true
if [[ -n "${PROCEDURAL_NATURAL_YIELD_SCAFFOLD:-}" ]]; then
  echo "Y3 invalid: constrained-yield scaffold is still enabled" >&2
  exit 1
fi

PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  "$root/scripts/run_qwen35_27b_natural_yield_sdpo_gate_battery_v1.sh" \
  "$base_model" "$base_label"
PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  "$root/scripts/run_qwen35_27b_natural_yield_sdpo_gate_battery_v1.sh" \
  "$candidate_model" "$candidate_label"

report=$evaluation_root/Y3_COMPARISON.json
"$root/.venv/bin/python" -m scripts.compare_natural_yield_sdpo_gates_v1 \
  "$evaluation_root" "$base_label" "$candidate_label" --output "$report"

"$root/.venv/bin/python" - "$report" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    report = json.load(handle)
decision = report["decision"]
if not decision["eligible_for_independent_replication"]:
    raise SystemExit("Y3 rejected: candidate does not improve native yield while retaining harness controls")
print("Y3_REPLICATION_ELIGIBLE_NOT_PROMOTED")
PY
