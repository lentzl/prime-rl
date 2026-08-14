#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PRIME_AGENT_OPD_STUDENT=Qwen/Qwen3.5-9B
export PRIME_AGENT_OPD_STUDENT_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export PRIME_AGENT_OPD_OUTPUT=${PRIME_AGENT_OPD_OUTPUT:-/ephemeral/subagent-rung/outputs/329-qwen35-9b-prime-agent-mastery-opd-smoke}
export PRIME_AGENT_OPD_TEACHER_LOG=${PRIME_AGENT_OPD_TEACHER_LOG:-/ephemeral/subagent-rung/logs/329-qwen35-27b-opd-teacher.log}

exec "$root/scripts/run_qwen35_prime_agent_opd_smoke.sh" "$@"
