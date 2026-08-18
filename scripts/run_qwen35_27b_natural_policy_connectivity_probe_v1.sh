#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
template=${NATURAL_POLICY_PROBE_CONFIG:-$experiment/natural-policy-connectivity-probe.toml}
rung=${NATURAL_POLICY_RUNG:-natural_n1}
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-r7-$rung-connectivity-probe-r1}
evaluation_root=${PROCEDURAL_HARNESS_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-natural-policy-ramp-v1}
start_index=${NATURAL_POLICY_PROBE_START_INDEX:-3100000}

case "$rung" in
  natural_n1|natural_n2) ;;
  *) echo "unknown natural-policy rung: $rung" >&2; exit 1 ;;
esac
if [[ ! -f "$template" ]]; then
  echo "natural-policy connectivity config does not exist: $template" >&2
  exit 1
fi
if [[ ! "$start_index" =~ ^[0-9]+$ ]]; then
  echo "natural-policy probe start index must be non-negative: $start_index" >&2
  exit 1
fi
"$root/scripts/build_prime_agent_runtime_image_v1.sh" >/dev/null

resolved_config=$(mktemp --suffix=.toml)
trap 'rm -f "$resolved_config"' EXIT
"$root/.venv/bin/python" - "$template" "$resolved_config" "$rung" "$start_index" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
for pattern, replacement, description in (
    (r'^curriculum_rung = "[^"]+"$', f'curriculum_rung = "{sys.argv[3]}"', "curriculum_rung"),
    (r"^start_index = [0-9]+$", f"start_index = {sys.argv[4]}", "start_index"),
):
    source, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"connectivity config must contain one {description}")
Path(sys.argv[2]).write_text(source)
PY

PROCEDURAL_HARNESS_ADMISSION_CONFIG=$resolved_config \
PROCEDURAL_HARNESS_OUTPUT_ROOT=$evaluation_root \
  "$root/scripts/run_qwen35_27b_procedural_harness_master_admission_v1.sh" \
  "$model" "$label"

summary=$evaluation_root/$label/train-admission/SUMMARY.json
"$root/.venv/bin/python" - "$summary" "$rung" <<'PY'
import json
import sys

from scripts.summarize_procedural_harness_master_v1 import (
    classify_natural_connectivity_probe,
)

with open(sys.argv[1]) as handle:
    report = json.load(handle)
print(classify_natural_connectivity_probe(report, sys.argv[2]))
PY
