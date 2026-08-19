#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export NATURAL_YIELD_SDPO_AUDIT_RUN_NAME=${NATURAL_YIELD_SDPO_AUDIT_RUN_NAME:-teacher-distribution-audit-v1}
export NATURAL_YIELD_SDPO_AUDIT_CONFIG=${NATURAL_YIELD_SDPO_AUDIT_CONFIG:-$root/experiments/qwen35-27b-procedural-harness-master-v1/natural-yield-sdpo-teacher-admission.toml}
export NATURAL_YIELD_SDPO_AUDIT_VALIDATOR_MODULE=scripts.audit_natural_yield_sdpo_teacher_distribution_v1

exec "$root/scripts/run_qwen35_27b_natural_yield_sdpo_zero_lr_v1.sh"
