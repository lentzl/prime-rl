#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revision=${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}
base_model=Qwen/Qwen3.5-27B
base_mastery=/ephemeral/evals/qwen35-27b-prime-agent-mastery-v2/untouched-base/SUMMARY.json
base_resilience=/ephemeral/evals/qwen35-27b-prime-agent-resilience-v1/untouched-base/SUMMARY.json
audit=/ephemeral/outputs/qwen35-27b-prime-agent-sdpo-v1/zero-lr-audit/AUDIT.json
update=/ephemeral/outputs/qwen35-27b-prime-agent-sdpo-v1/minimum-update/UPDATE.json
weights=/ephemeral/outputs/qwen35-27b-prime-agent-sdpo-v1/minimum-update/weights/step_1
candidate_mastery=/ephemeral/evals/qwen35-27b-prime-agent-mastery-v2/minimum-update/SUMMARY.json
candidate_resilience=/ephemeral/evals/qwen35-27b-prime-agent-resilience-v1/minimum-update/SUMMARY.json
sequence_dir=/ephemeral/evals/qwen35-27b-prime-agent-teacher-candidate-v1
comparison=$sequence_dir/COMPARISON.json

cd "$root"
if [[ ! -s "$base_mastery" ]]; then
  echo "untouched mastery baseline is incomplete" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to start the teacher-candidate sequence while a GPU process is active" >&2
  exit 1
fi
for executable in .venv/bin/python .venv/bin/ruff .venv/bin/pytest; do
  if [[ ! -x "$executable" ]]; then
    echo "required preflight executable is missing: $executable" >&2
    exit 1
  fi
done

.venv/bin/ruff check \
  scripts/compare_prime_agent_teacher_candidate_v1.py \
  scripts/validate_prime_agent_sdpo_zero_lr_audit_v1.py \
  scripts/validate_prime_agent_sdpo_minimum_update_v1.py \
  tests/unit/test_compare_prime_agent_teacher_candidate_v1.py \
  tests/unit/test_prime_agent_sdpo_audit.py \
  tests/unit/test_validate_prime_agent_sdpo_zero_lr_audit_v1.py \
  tests/unit/test_prime_agent_sdpo_minimum_update.py \
  tests/unit/test_mastery_battery_v2.py \
  tests/unit/test_prime_agent_resilience_battery_v1.py \
  tests/unit/test_prime_agent_teacher_candidate_sequence_v1.py
.venv/bin/pytest -q \
  tests/unit/test_compare_prime_agent_teacher_candidate_v1.py \
  tests/unit/test_prime_agent_sdpo_audit.py \
  tests/unit/test_validate_prime_agent_sdpo_zero_lr_audit_v1.py \
  tests/unit/test_prime_agent_sdpo_minimum_update.py \
  tests/unit/test_mastery_battery_v2.py \
  tests/unit/test_prime_agent_resilience_battery_v1.py \
  tests/unit/test_prime_agent_teacher_candidate_sequence_v1.py \
  deps/verifiers/tests/v1/test_prime_agent_resilience.py

wait_for_idle_gpus() {
  local attempts=0
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    attempts=$((attempts + 1))
    if ((attempts > 60)); then
      echo "GPU processes did not exit within ten minutes" >&2
      return 1
    fi
    sleep 10
  done
}

if [[ ! -s "$base_resilience" ]]; then
  PRIME_MASTERY_OUTPUT_ROOT=/ephemeral/evals/qwen35-27b-prime-agent-resilience-v1 \
    EVAL_DRIVER=scripts/run_qwen35_27b_prime_agent_resilience_v1.sh \
    EVAL_EXPERIMENT_DIR=experiments/qwen35-27b-prime-agent-resilience-v1 \
    bash scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh \
    "$base_model" untouched-base "$revision"
fi
wait_for_idle_gpus

if [[ ! -s "$audit" ]]; then
  bash scripts/run_qwen35_27b_prime_agent_sdpo_zero_lr_audit_v1.sh
fi
.venv/bin/python - "$audit" "$revision" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
if (
    report.get("verdict") != "pass"
    or report.get("mechanism") != "mixed-feedback-conditioned-sdpo-grpo-zero-lr"
    or report.get("expected_revision") != sys.argv[2]
    or report.get("model_artifacts_written") is not False
):
    raise SystemExit("zero-LR audit report is not a matching pass")
PY
wait_for_idle_gpus

if [[ ! -s "$update" ]]; then
  SDPO_MINIMUM_UPDATE_DRY_RUN=true \
    bash scripts/run_qwen35_27b_prime_agent_sdpo_minimum_update_v1.sh
  bash scripts/run_qwen35_27b_prime_agent_sdpo_minimum_update_v1.sh
fi
.venv/bin/python scripts/validate_prime_agent_sdpo_minimum_update_v1.py \
  "$(dirname "$update")" \
  --expected-revision "$revision" \
  --audit-report "$audit" \
  >/dev/null
wait_for_idle_gpus

if [[ ! -s "$candidate_mastery" ]]; then
  bash scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh \
    "$weights" minimum-update "$revision"
fi
wait_for_idle_gpus

if [[ ! -s "$candidate_resilience" ]]; then
  PRIME_MASTERY_OUTPUT_ROOT=/ephemeral/evals/qwen35-27b-prime-agent-resilience-v1 \
    EVAL_DRIVER=scripts/run_qwen35_27b_prime_agent_resilience_v1.sh \
    EVAL_EXPERIMENT_DIR=experiments/qwen35-27b-prime-agent-resilience-v1 \
    bash scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh \
    "$weights" minimum-update "$revision"
fi

mkdir -p "$sequence_dir"
.venv/bin/python scripts/compare_prime_agent_teacher_candidate_v1.py \
  --base-mastery "$base_mastery" \
  --candidate-mastery "$candidate_mastery" \
  --base-resilience "$base_resilience" \
  --candidate-resilience "$candidate_resilience" \
  --output "$comparison" \
  >"$sequence_dir/COMPARISON.txt"

{
  printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'prime_rl_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'verifiers_commit=%s\n' "$(git -C deps/verifiers rev-parse HEAD)"
  printf 'base_revision=%s\n' "$revision"
  printf 'base_mastery=%s\n' "$base_mastery"
  printf 'base_resilience=%s\n' "$base_resilience"
  printf 'audit=%s\n' "$audit"
  printf 'update=%s\n' "$update"
  printf 'candidate_mastery=%s\n' "$candidate_mastery"
  printf 'candidate_resilience=%s\n' "$candidate_resilience"
  printf 'comparison=%s\n' "$comparison"
} >"$sequence_dir/COMPLETE.txt"

echo "teacher-candidate sequence completed: $sequence_dir/COMPLETE.txt"
