#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen35-27b-procedural-harness-master-v1
template=${HARNESS_ACTION_ADMISSION_CONFIG:-$experiment/harness-action-admission.toml}
rung=${1:-atomic_state}
model=${2:-Qwen/Qwen3.5-27B}
label=${3:-untouched-$rung}
evaluation_root=${PROCEDURAL_HARNESS_OUTPUT_ROOT:-${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-procedural-harness-action-ramp-v1}}
start_index=${HARNESS_ACTION_ADMISSION_START_INDEX:-900000}
record_causal_feedback=${HARNESS_ACTION_RECORD_CAUSAL_FEEDBACK:-false}

case "$rung" in
  atomic_state|atomic_send|atomic_child_request|atomic_followup|atomic_parallel) ;;
  *) echo "unknown harness-action rung: $rung" >&2; exit 1 ;;
esac
if [[ ! -f "$template" ]]; then
  echo "harness-action admission config does not exist: $template" >&2
  exit 1
fi
if [[ ! "$start_index" =~ ^[0-9]+$ ]]; then
  echo "harness-action admission start index must be non-negative: $start_index" >&2
  exit 1
fi
if [[ "$record_causal_feedback" != true && "$record_causal_feedback" != false ]]; then
  echo "HARNESS_ACTION_RECORD_CAUSAL_FEEDBACK must be true or false: $record_causal_feedback" >&2
  exit 1
fi
"$root/scripts/build_prime_agent_runtime_image_v1.sh" >/dev/null

resolved_config=$(mktemp --suffix=.toml)
trap 'rm -f "$resolved_config"' EXIT
"$root/.venv/bin/python" - "$template" "$resolved_config" "$rung" "$start_index" "$record_causal_feedback" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
rendered, replacements = re.subn(
    r'^curriculum_rung = "[^"]+"$',
    f'curriculum_rung = "{sys.argv[3]}"',
    source,
    count=1,
    flags=re.MULTILINE,
)
if replacements != 1:
    raise SystemExit("admission config must contain one curriculum_rung")
rendered, replacements = re.subn(
    r"^start_index = [0-9]+$",
    f"start_index = {sys.argv[4]}",
    rendered,
    count=1,
    flags=re.MULTILINE,
)
if replacements != 1:
    raise SystemExit("admission config must contain one start_index")
rendered, replacements = re.subn(
    r"^record_causal_feedback = (?:true|false)$",
    f"record_causal_feedback = {sys.argv[5]}",
    rendered,
    count=1,
    flags=re.MULTILINE,
)
if replacements != 1:
    raise SystemExit("admission config must contain one record_causal_feedback")
Path(sys.argv[2]).write_text(rendered)
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
    classify_curriculum_rung_admission,
)

with open(sys.argv[1]) as handle:
    report = json.load(handle)
print(classify_curriculum_rung_admission(report, sys.argv[2]))
PY
