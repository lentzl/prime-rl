#!/usr/bin/env bash
set -euo pipefail

ROOT=${RUNG_ROOT:-/ephemeral/subagent-rung}
REPORT=${BOOTSTRAP_AUDIT_REPORT:-$ROOT/data/281-qwen35-27b-prime-agent-teacher-bootstrap-audit.json}
POLL_SECONDS=${POLL_SECONDS:-60}

wait_for_rollouts() {
  local log=$1
  local expected=$2
  local completed=0
  while (( completed < expected )); do
    if [[ -f "$log" ]]; then
      completed=$(grep -c "rollout done:" "$log" || true)
    fi
    printf '%s: %d/%d rollouts complete\n' "$(basename "$log")" "$completed" "$expected"
    (( completed >= expected )) || sleep "$POLL_SECONDS"
  done
}

wait_for_rollouts "$ROOT/logs/278-qwen35-27b-mastery-child-teacher-collection-base-r1.log" 128
wait_for_rollouts "$ROOT/logs/279-qwen35-27b-mastery-coordinator-teacher-collection-base-r1.log" 128
wait_for_rollouts "$ROOT/logs/280-qwen35-27b-mastery-guided-communication-collection-base-r1.log" 160

mkdir -p "$(dirname "$REPORT")"
set +e
AUDIT_ONLY=1 bash scripts/build_prime_agent_teacher_bootstrap.sh >"$REPORT"
status=$?
set -e
printf '%d\n' "$status" >"$REPORT.status"
exit "$status"
