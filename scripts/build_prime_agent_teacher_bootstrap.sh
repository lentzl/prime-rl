#!/usr/bin/env bash
set -euo pipefail

ROOT=${RUNG_ROOT:-/ephemeral/subagent-rung}
OUTPUT=${BOOTSTRAP_OUTPUT:-$ROOT/data/281-qwen35-27b-prime-agent-teacher-bootstrap}
CHILD_RUN=${CHILD_RUN:-$ROOT/evals/278-qwen35-27b-mastery-child-teacher-collection/base-r1}
COORDINATOR_RUN=${COORDINATOR_RUN:-$ROOT/evals/279-qwen35-27b-mastery-coordinator-teacher-collection/base-r1}
COMMUNICATION_RUN=${COMMUNICATION_RUN:-$ROOT/evals/280-qwen35-27b-mastery-guided-communication-collection/base-r1}

if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite existing bootstrap output: $OUTPUT" >&2
  exit 1
fi

ownership_families=(
  json_sum
  csv_amount_total
  text_keyword_count
  markdown_heading_count
  log_error_count
  python_def_count
  json_max_value
  sha256_prefix
)
communication_families=(direct single parallel followup handshake)

requirements=()
for ownership in child coordinator; do
  for family in "${ownership_families[@]}"; do
    requirements+=(--require-count "ownership.$ownership.family.$family=1")
  done
done
for family in "${communication_families[@]}"; do
  requirements+=(--require-count "family.$family=4")
done

python scripts/export_prime_agent_teacher_bootstrap.py \
  --ownership-run "$CHILD_RUN" \
  --ownership-run "$COORDINATOR_RUN" \
  --communication-run "$COMMUNICATION_RUN" \
  "${requirements[@]}" \
  --output-dir "$OUTPUT"
