#!/usr/bin/env bash
set -euo pipefail

root=${RUNG_ROOT:-/ephemeral/subagent-rung}
checkout=${MASTERY_CHECKOUT:-/home/ubuntu/prime-rl-mastery-launch}
followup_session=${FOLLOWUP_SESSION:-mastery-314-literal-safe-followup}
followup_run=${FOLLOWUP_RUN:-$root/evals/314-qwen35-27b-mastery-literal-safe-followup-supplement/base-r1}
audit_report=${BOOTSTRAP_AUDIT_REPORT:-$root/data/285-qwen35-27b-prime-agent-teacher-final-audit.json}
dataset=${BOOTSTRAP_OUTPUT:-$root/data/281-qwen35-27b-prime-agent-teacher-bootstrap}
poll_seconds=${POLL_SECONDS:-30}

child_runs=${CHILD_RUNS:-$root/evals/278-qwen35-27b-mastery-child-teacher-collection/base-r1:$root/evals/290-qwen35-27b-mastery-child-ownership-supplement-r2/base-r1}
coordinator_runs=${COORDINATOR_RUNS:-$root/evals/279-qwen35-27b-mastery-coordinator-teacher-collection/base-r1:$root/evals/288-qwen35-27b-mastery-guided-coordinator-ownership-supplement/base-r1}
communication_runs=${COMMUNICATION_RUNS:-$root/evals/280-qwen35-27b-mastery-guided-communication-collection/base-r1:$root/evals/292-qwen35-27b-mastery-corrected-parallel-supplement/base-r1:$root/evals/308-qwen35-27b-mastery-evidence-gated-handshake-supplement/base-r1:$followup_run}

count_admitted() {
  python3 - "$1/traces.jsonl" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
count = 0
if path.exists():
    for line in path.open():
        metrics = json.loads(line)["traces"][0].get("metrics", {})
        count += (
            metrics.get("answer_accuracy") == 1
            and metrics.get("clean_protocol_aligned") == 1
            and metrics.get("natural_followup_causal") == 1
            and metrics.get("bidirectional_control") == 1
        )
print(count)
PY
}

while :; do
  admitted=$(count_admitted "$followup_run")
  printf 'follow-up corpus: %d/4 clean admitted traces\n' "$admitted"
  if (( admitted >= 4 )); then
    tmux send-keys -t "$followup_session" C-c 2>/dev/null || true
    sleep 4
    tmux kill-session -t "$followup_session" 2>/dev/null || true
    break
  fi
  if ! tmux has-session -t "$followup_session" 2>/dev/null; then
    echo "follow-up collection ended before reaching four clean traces" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done

cd "$checkout"
mkdir -p "$(dirname "$audit_report")"
CHILD_RUNS="$child_runs" \
COORDINATOR_RUNS="$coordinator_runs" \
COMMUNICATION_RUNS="$communication_runs" \
AUDIT_ONLY=1 \
bash scripts/build_prime_agent_teacher_bootstrap.sh >"$audit_report"

if [[ -e "$dataset" ]]; then
  echo "refusing to overwrite existing bootstrap dataset: $dataset" >&2
  exit 1
fi
CHILD_RUNS="$child_runs" \
COORDINATOR_RUNS="$coordinator_runs" \
COMMUNICATION_RUNS="$communication_runs" \
BOOTSTRAP_OUTPUT="$dataset" \
bash scripts/build_prime_agent_teacher_bootstrap.sh

for session in mastery-inference mastery-inference-secondary; do
  tmux send-keys -t "$session" C-c 2>/dev/null || true
done
for _ in $(seq 1 60); do
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
  sleep 5
done
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU processes did not quiesce after stopping inference" >&2
  exit 1
fi

BOOTSTRAP_DATASET="$dataset" exec bash scripts/run_prime_agent_teacher_bootstrap_online.sh
