#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PRIME_AGENT_OPD_STUDENT=Qwen/Qwen3.5-2B
export PRIME_AGENT_OPD_STUDENT_REVISION=15852e8c16360a2fea060d615a32b45270f8a8fc
export PRIME_AGENT_OPD_OUTPUT=${PRIME_AGENT_OPD_OUTPUT:-/ephemeral/subagent-rung/outputs/qwen35-2b-prime-agent-mastery-opd-smoke}
export PRIME_AGENT_OPD_TEACHER_LOG=${PRIME_AGENT_OPD_TEACHER_LOG:-/ephemeral/subagent-rung/logs/qwen35-27b-opd-teacher-for-2b.log}

exec "$root/scripts/run_qwen35_prime_agent_opd_smoke.sh" "$@"
