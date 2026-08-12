#!/usr/bin/env bash
set -euo pipefail

ROOT=${RUNG_ROOT:-/ephemeral/subagent-rung}
REPORT=${BOOTSTRAP_AUDIT_REPORT:-$ROOT/data/281-qwen35-27b-prime-agent-teacher-bootstrap-audit.json}
POLL_SECONDS=${POLL_SECONDS:-60}

wait_for_episodes() {
  local run_dir=$1
  local expected=$2
  local completed=0
  local traces="$run_dir/traces.jsonl"
  while (( completed < expected )); do
    if [[ -f "$traces" ]]; then
      completed=$(wc -l <"$traces")
    fi
    printf '%s: %d/%d persisted episodes\n' "$(basename "$(dirname "$run_dir")")" "$completed" "$expected"
    (( completed >= expected )) || sleep "$POLL_SECONDS"
  done
}

wait_for_episodes "$ROOT/evals/278-qwen35-27b-mastery-child-teacher-collection/base-r1" 128
wait_for_episodes "$ROOT/evals/279-qwen35-27b-mastery-coordinator-teacher-collection/base-r1" 128
wait_for_episodes "$ROOT/evals/280-qwen35-27b-mastery-guided-communication-collection/base-r1" 160

mkdir -p "$(dirname "$REPORT")"
set +e
AUDIT_ONLY=1 bash scripts/build_prime_agent_teacher_bootstrap.sh >"$REPORT"
status=$?
set -e
printf '%d\n' "$status" >"$REPORT.status"
exit "$status"
