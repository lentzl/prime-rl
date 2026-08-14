#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PRIME_AGENT_OPD_STUDENT=Qwen/Qwen3.5-4B
export PRIME_AGENT_OPD_STUDENT_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export PRIME_AGENT_OPD_OUTPUT=${PRIME_AGENT_OPD_OUTPUT:-/ephemeral/subagent-rung/outputs/qwen35-4b-prime-agent-mastery-opd-smoke}
export PRIME_AGENT_OPD_TEACHER_LOG=${PRIME_AGENT_OPD_TEACHER_LOG:-/ephemeral/subagent-rung/logs/qwen35-27b-opd-teacher-for-4b.log}

exec "$root/scripts/run_qwen35_prime_agent_opd_smoke.sh" "$@"
