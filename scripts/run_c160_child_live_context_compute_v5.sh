#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
harvest_a=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/c158-c160-step10-compute-report-curriculum-9424710-n24/natural_n1a/traces.jsonl
harvest_b=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/c158-c160-step10-compute-report-curriculum-resume2-9424723-n11/natural_n1a/traces.jsonl
harvest_c=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/c158-c160-step10-compute-report-curriculum-log-error-9424736-n1/natural_n1a/traces.jsonl
corpus=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c160-child-live-context-compute-v5
candidate=/home/ubuntu/rlm/outputs/q35-2b-recursive-coordinator-return-v1/c160-child-live-context-compute-v5/weights/step_8
coordinator=/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1/grpo-auto-000158-coordinator-e0d3-uncapped-yield-exact-child-9415800/weights/step_1
start=9424900
label=c158-c160-child-live-context-compute-v5-step8-9424900-n6
bootstrap=/home/ubuntu/rlm/artifacts/q35-2b-recursive-coordinator-return-v1/c-return-9424900-n6-root-action-bootstrap.json

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -e "$corpus" ]]; then
  python scripts/build_q35_2b_recursive_return_trace_sft_v1.py \
    --forced-return-traces "$harvest_a" \
    --forced-return-traces "$harvest_b" \
    --forced-return-traces "$harvest_c" \
    --child-only --scaffolded-compute-actions --leaf-reporter-contract \
    --return-repeats 4 --minimum-return-traces 16 --output-dir "$corpus"
fi
jq -e '
  .status == "complete"
  and .accepted_return_trajectories >= 16
  and .leaf_reporter_contract == true
  and .replay_anchor_rows == 0
  and .recursive_return_rows == .row_count
  and ((.resource_family_counts | keys | sort) ==
    (["csv_total", "json_max", "json_sum", "log_error", "md_h2", "python_defs", "word_count"] | sort))
' "$corpus/MANIFEST.json" >/dev/null
expected_corpus_sha=$(jq -er '.dataset.sha256' "$corpus/MANIFEST.json")
actual_corpus_sha=$(sha256sum "$corpus/train.parquet" | awk '{print $1}')
if [[ "$actual_corpus_sha" != "$expected_corpus_sha" ]]; then
  echo "live-context compute corpus checksum mismatch" >&2
  exit 1
fi
python - "$corpus/train.parquet" <<'PY'
import ast
import json
import sys
from datasets import Dataset
sys.path.insert(0, "scripts")
from dual_policy_openai_proxy_v1 import LEAF_REPORTER_CONTRACT

rows = list(Dataset.from_parquet(sys.argv[1]))
assert rows
for row in rows:
    messages = row["messages"]
    assert [message["role"] for message in messages[:3]] == ["system", "user", "user"]
    assert messages[0]["content"] == LEAF_REPORTER_CONTRACT
    assistant = next(message for message in messages if message["role"] == "assistant")
    code = json.loads(assistant["tool_calls"][0]["arguments"])["code"]
    tree = ast.parse(code)
    assert "INLINE_EVIDENCE" in code
    assert "receiver_role='parent'" in code
    assert row["expected_result"] not in {
        str(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    }
print(f"audited {len(rows)} live-context, answer-free compute targets")
PY
if [[ ! -f "$candidate/STABLE" || ! -f "$candidate/model.safetensors" ]]; then
  sft @ experiments/qwen35-2b-recursive-coordinator-return-v1/c160-child-live-context-compute-v5.toml
fi
test -f "$candidate/STABLE" -a -f "$candidate/model.safetensors"
python scripts/build_q35_2b_environment_bootstrap_context_v1.py \
  --output "$bootstrap" --axis natural_n1a:"$start" --tasks-per-axis 6 \
  --leak-level action_scaffold

QWEN38_QUALIFICATION_CONFIG="$root/experiments/qwen35-2b-recursive-coordinator-return-v1/qualification-template.toml" \
QWEN38_QUALIFICATION_OUTPUT_ROOT=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1 \
QWEN38_QUALIFICATION_AXES=natural_n1a \
QWEN38_QUALIFICATION_NUM_TASKS=6 \
QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
QWEN38_QUALIFICATION_START_INDEX="$start" \
QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH="$bootstrap" \
PROCEDURAL_INTERACTION_CURRICULUM=e0c4_recursive_coordinator_return \
PROCEDURAL_NATURAL_YIELD_SCAFFOLD=1 \
DUAL_EXTERNAL_MODEL=q35-2b-recursive-coordinator-return \
DUAL_ROOT_COORDINATOR_CONTRACT=1 \
DUAL_LEAF_REPORTER_CONTRACT=1 \
DUAL_LEAF_INLINE_EVIDENCE=1 \
UV_BIN=/home/ubuntu/.local/bin/uv \
scripts/run_q35_2b_dual_policy_mastery_v1.sh \
  "$coordinator" "$candidate" "$label" candidate-local

traces=/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1/$label/natural_n1a/traces.jsonl
jq -s '{
  episodes: length,
  errors: ([.[] | select((.errors | length) > 0)] | length),
  qualifying: ([.[] | select(
    .traces[0].rewards.harness_score.score == 1
    and .traces[0].metrics.child_action_completed == 1
    and .traces[0].stop_condition == "user_closed"
  )] | length),
  admission_floor: 4,
  admitted: (([.[] | select(
    .traces[0].rewards.harness_score.score == 1
    and .traces[0].metrics.child_action_completed == 1
    and .traces[0].stop_condition == "user_closed"
  )] | length) >= 4)
}' "$traces"
