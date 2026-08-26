#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
role=${1:?role required: coordinator or child}
model_path=${2:?absolute dense model checkpoint required}
anchor_model_path=${3:?absolute frozen counterpart checkpoint required}
run_name=${4:?unique run name required}
phase=${5:-e0d3_uncapped_yield_exact_child}
start_index=${6:-9100000}
task_count=${7:-64}
template=${Q35_2B_ROLE_GRPO_TEMPLATE:-$root/experiments/qwen35-2b-self-bootstrap-dual-dense-v1/role-shaped-grpo.toml}
experiment_dir=${Q35_2B_ROLE_GRPO_EXPERIMENT_DIR:-$root/experiments/qwen35-2b-self-bootstrap-dual-dense-v1/grpo-runs}
artifact_root=${Q35_2B_ROLE_GRPO_ARTIFACT_ROOT:-/home/ubuntu/rlm/artifacts/q35-2b-self-bootstrap-dual-dense-grpo-v1}
output_root=${Q35_2B_ROLE_GRPO_OUTPUT_ROOT:-/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1}
uv_bin=${UV_BIN:-/home/ubuntu/.local/bin/uv}
learning_rate=${Q35_2B_ROLE_GRPO_LR:-1e-6}
dry_run=${Q35_2B_ROLE_GRPO_DRY_RUN:-false}
master_seed=20260824

case "$role" in
  # Keep the frozen child reliable while collecting coordinator trajectories.
  # The child owns the hidden evidence, so this exposes the exact send only in
  # its private context; coordinator tokens still have to learn how to resume
  # from and use the delivered report.
  coordinator) scope=root; leak_coordinator_exact_action=true; leak_child_exact_action=true; strip_child_tool_choice=false; strip_coordinator_tool_choice=false; enable_thinking=false ;;
  child) scope=non_root; leak_coordinator_exact_action=true; leak_child_exact_action=true; strip_child_tool_choice=true; strip_coordinator_tool_choice=false; enable_thinking=true ;;
  *) echo "role must be coordinator or child: $role" >&2; exit 1 ;;
esac
case "$role:$phase" in
  coordinator:e0d3_uncapped_yield_exact_child) ;;
  child:e0_full_actions|child:e0c_natural_child|child:e0c2_natural_child_no_template|child:e0c25_inline_evidence|child:e0c275_inline_location|child:e0c28_inline_only|child:e0c29_evidence_available|child:e0c3_natural_child_minimal) ;;
  *) echo "unsupported role/phase pairing for stabilized role GRPO: $role:$phase" >&2; exit 1 ;;
esac
bootstrap_leak_level=action_scaffold
max_completion_tokens=2048
agent_max_turns=16
agent_max_output_tokens=16384
agent_max_total_tokens=65536
autonomous_max_tokens=65536
if [[ "$role:$phase" == child:e0_full_actions ]]; then
  bootstrap_leak_level=solution_replay
  max_completion_tokens=1024
  agent_max_turns=8
  agent_max_output_tokens=8192
  agent_max_total_tokens=32768
  autonomous_max_tokens=32768
fi
if [[ ! "$run_name" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "run name must be a stable lowercase label: $run_name" >&2
  exit 1
fi
if [[ ! "$start_index" =~ ^[0-9]+$ || ! "$task_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "start index and task count must be non-negative/positive integers" >&2
  exit 1
fi
for checkpoint in "$model_path" "$anchor_model_path"; do
  if [[ "$checkpoint" != /* || ! -f "$checkpoint/STABLE" || ! -f "$checkpoint/model.safetensors" ]]; then
    echo "role checkpoint is not an absolute complete dense checkpoint: $checkpoint" >&2
    exit 1
  fi
done
if [[ ! -f "$template" || ! -x "$uv_bin" ]]; then
  echo "role-GRPO template or uv executable is unavailable" >&2
  exit 1
fi
if [[ "$dry_run" != true && -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch role GRPO while another GPU process is active" >&2
  exit 1
fi

cd "$root"
export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
mkdir -p "$experiment_dir" "$artifact_root"
resolved=$experiment_dir/$run_name.toml
bootstrap=$artifact_root/$run_name-action-scaffold-bootstrap.json
receipt=$experiment_dir/$run_name-receipt.json
attempt_receipt=$experiment_dir/$run_name-attempt.json
run_output=$output_root/$run_name
role_router_state=$experiment_dir/$run_name-role-router
routing_audit=$role_router_state/ROUTING_AUDIT.jsonl
for target in "$resolved" "$bootstrap" "$receipt" "$attempt_receipt" "$run_output" "$role_router_state"; do
  if [[ -e "$target" ]]; then
    echo "refusing duplicate role-GRPO target: $target" >&2
    exit 1
  fi
done

attempt_stage=preflight
attempt_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
write_attempt_receipt() {
  local exit_code=$1
  local status=failed
  if [[ "$attempt_stage" == complete && "$exit_code" -eq 0 ]]; then
    status=complete
  elif [[ "$attempt_stage" == dry_run_complete && "$exit_code" -eq 0 ]]; then
    status=dry_run_complete
  elif [[ "$attempt_stage" == no_update ]]; then
    status=no_update
  fi
  "$uv_bin" run --frozen --no-sync python - \
    "$attempt_receipt" "$run_name" "$role" "$scope" "$phase" \
    "$start_index" "$task_count" "$attempt_started_at" "$attempt_stage" \
    "$status" "$exit_code" "$resolved" "$bootstrap" "$run_output" \
    "$receipt" "$routing_audit" <<'PY' || true
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    destination,
    run_name,
    role,
    scope,
    phase,
    start_index,
    task_count,
    started_at,
    stage,
    status,
    exit_code,
    config,
    bootstrap,
    output,
    receipt,
    routing_audit,
) = sys.argv[1:]
payload = {
    "schema_version": "qwen35-2b-role-grpo-attempt/v1",
    "run_name": run_name,
    "role": role,
    "sampled_session_scope": scope,
    "coordinator_action_leak": True,
    "child_action_leak": True,
    "first_action_sampling": "synthetic_exact_spawn" if role == "coordinator" else "masked_frozen_anchor",
    "child_tool_choice_stripped": role == "child",
    "coordinator_tool_choice_stripped": False,
    "env_server_max_concurrent": 1 if role == "coordinator" else 2,
    "enable_thinking": role == "child",
    "bootstrap_leak_level": "solution_replay" if phase == "e0_full_actions" else "action_scaffold",
    "early_rung_bounded": phase == "e0_full_actions",
    "phase": phase,
    "task_bank": {"start_index": int(start_index), "count": int(task_count)},
    "started_at_utc": started_at,
    "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "stage": stage,
    "status": status,
    "exit_code": int(exit_code),
    "config_path": config,
    "bootstrap_path": bootstrap,
    "output_path": output,
    "success_receipt_path": receipt if Path(receipt).is_file() else None,
    "routing_audit_path": routing_audit if Path(routing_audit).is_file() else None,
    "promotion_minimum": 4,
    "acceptance_floor_relaxed": False,
}
Path(destination).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}
trap 'attempt_exit=$?; trap - EXIT; write_attempt_receipt "$attempt_exit"; exit "$attempt_exit"' EXIT

"$uv_bin" run --frozen --no-sync scripts/build_q35_2b_environment_bootstrap_context_v1.py \
  --output "$bootstrap" \
  --axis "natural_n1a:$start_index" \
  --tasks-per-axis "$task_count" \
  --master-seed "$master_seed" \
  --leak-level "$bootstrap_leak_level"

"$uv_bin" run --frozen --no-sync python - \
  "$template" "$resolved" "$role" "$scope" "$model_path" "$anchor_model_path" \
  "$run_name" "$bootstrap" "$start_index" "$task_count" "$output_root" \
  "$learning_rate" "$role_router_state" "$routing_audit" \
  "$leak_coordinator_exact_action" "$strip_child_tool_choice" \
  "$strip_coordinator_tool_choice" "$enable_thinking" \
  "$max_completion_tokens" "$agent_max_turns" "$agent_max_output_tokens" \
  "$agent_max_total_tokens" "$autonomous_max_tokens" <<'PY'
import math
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
learning_rate = float(sys.argv[12])
if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise SystemExit("learning rate must be positive and finite")
replacements = (
    (r'^output_dir = ".*"$', f'output_dir = "{sys.argv[11]}"', 1),
    (r'^name = "role-grpo-shadow"$', f'name = "{sys.argv[7]}"', 1),
    (r'^dir = "role-grpo-shadow"$', f'dir = "{sys.argv[7]}"', 1),
    (r'^name = "Qwen/Qwen3.5-2B"$', f'name = "{sys.argv[5]}"', 1),
    (r'^sampled_session_scope = "root"$', f'sampled_session_scope = "{sys.argv[4]}"', 2),
    (r'^policy_role = "coordinator"$', f'policy_role = "{sys.argv[3]}"', 1),
    (
        r'^leak_coordinator_exact_action = false$',
        f'leak_coordinator_exact_action = {sys.argv[15]}',
        1,
    ),
    (
        r'^leak_child_exact_action = false$',
        'leak_child_exact_action = true',
        1,
    ),
    (
        r'^strip_child_tool_choice = false$',
        f'strip_child_tool_choice = {sys.argv[16]}',
        1,
    ),
    (
        r'^strip_coordinator_tool_choice = false$',
        f'strip_coordinator_tool_choice = {sys.argv[17]}',
        1,
    ),
    (r'^enable_thinking = false$', f'enable_thinking = {sys.argv[18]}', 1),
    (r'^max_completion_tokens = 2048$', f'max_completion_tokens = {sys.argv[19]}', 1),
    (r'^max_turns = 16$', f'max_turns = {sys.argv[20]}', 1),
    (r'^max_output_tokens = 16384$', f'max_output_tokens = {sys.argv[21]}', 1),
    (r'^max_total_tokens = 65536$', f'max_total_tokens = {sys.argv[22]}', 1),
    (r'^autonomous_max_turns = 16$', f'autonomous_max_turns = {sys.argv[20]}', 1),
    (r'^autonomous_max_tokens = 65536$', f'autonomous_max_tokens = {sys.argv[23]}', 1),
    (
        r'^max_concurrent = 1$',
        f'max_concurrent = {1 if sys.argv[3] == "coordinator" else 2}',
        1,
    ),
    (r'^anchor_model = "/tmp/q35-2b-role-grpo-anchor"$', f'anchor_model = "{sys.argv[6]}"', 1),
    (r'^state_dir = "/tmp/q35-2b-role-grpo-router"$', f'state_dir = "{sys.argv[13]}"', 1),
    (
        r'^audit_log = "/tmp/q35-2b-role-grpo-routing-audit.jsonl"$',
        f'audit_log = "{sys.argv[14]}"',
        1,
    ),
    (r'^lr = 1e-6$', f'lr = {learning_rate:.12g}', 1),
    (r'^count = 64$', f'count = {sys.argv[10]}', 1),
    (r'^start_index = 9100000$', f'start_index = {sys.argv[9]}', 1),
    (
        r'^privileged_bootstrap_path = "/tmp/q35-2b-role-grpo-bootstrap.json"$',
        f'privileged_bootstrap_path = "{sys.argv[8]}"',
        1,
    ),
)
for pattern, replacement, expected in replacements:
    source, count = re.subn(pattern, replacement, source, count=expected, flags=re.MULTILINE)
    if count != expected:
        raise SystemExit(f"template did not match {pattern!r} exactly {expected} time(s)")
Path(sys.argv[2]).write_text(source)
PY

"$uv_bin" run --frozen --no-sync scripts/validate_q35_2b_role_grpo_v1.py \
  "$resolved" --role "$role" --model-path "$model_path" \
  --anchor-model-path "$anchor_model_path" --run-name "$run_name" \
  --bootstrap-path "$bootstrap" --phase "$phase" \
  --start-index "$start_index" --task-count "$task_count"

# Exercise the exact environment materialization before any GPU process starts.
# This catches generator/bootstrap identity drift that a schema-only dry run cannot.
PROCEDURAL_INTERACTION_CURRICULUM="$phase" "$uv_bin" run --frozen --no-sync python - \
  "$bootstrap" "$start_index" "$task_count" "$master_seed" <<'PY'
import sys

from procedural_harness_master_v1.taskset import (
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTaskset,
)

tasks = ProceduralHarnessMasterTaskset(
    ProceduralHarnessMasterConfig(
        split="train_gen",
        count=int(sys.argv[3]),
        start_index=int(sys.argv[2]),
        master_seed=int(sys.argv[4]),
        curriculum_rung="natural_n1a",
        private_payload_mode="finding_card",
        privileged_bootstrap_path=sys.argv[1],
    )
).load()
if len(tasks) != int(sys.argv[3]):
    raise SystemExit("role-GRPO preflight materialized an incomplete task bank")
print({"materialized_tasks": len(tasks), "first": tasks[0].key, "last": tasks[-1].key})
PY

if [[ "$dry_run" == true ]]; then
  PROCEDURAL_INTERACTION_CURRICULUM="$phase" rl @ "$resolved" --dry-run
  attempt_stage=dry_run_complete
  echo "role-GRPO zero-update audit passed: $resolved"
  exit 0
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export PROCEDURAL_INTERACTION_CURRICULUM="$phase"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
for package in subagent_communication_v1 procedural_harness_master_v1; do
  "$uv_bin" pip install --python "$root/.venv/bin/python" --no-deps --editable \
    "$root/deps/verifiers/environments/$package" >/dev/null
done
attempt_stage=rl_running
rl @ "$resolved"
attempt_stage=rl_returned

weights=$run_output/weights/step_1
metrics=$run_output/metrics.jsonl
if [[ ! -f "$weights/STABLE" || ! -f "$weights/model.safetensors" || ! -f "$metrics" ]]; then
  attempt_stage=no_update
  echo "role-GRPO update did not produce a complete step-1 checkpoint" >&2
  exit 3
fi
"$uv_bin" run --frozen --no-sync python - \
  "$receipt" "$role" "$scope" "$phase" "$start_index" "$task_count" \
  "$model_path" "$anchor_model_path" "$weights" "$resolved" "$bootstrap" \
  "$metrics" "$routing_audit" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

receipt, role, scope, phase = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
source, anchor, output = Path(sys.argv[7]), Path(sys.argv[8]), Path(sys.argv[9])
payload = {
    "schema_version": "qwen35-2b-role-grpo-update/v1",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "role": role,
    "sampled_session_scope": scope,
    "coordinator_action_leak": True,
    "child_action_leak": True,
    "first_action_sampling": "synthetic_exact_spawn" if role == "coordinator" else "masked_frozen_anchor",
    "phase": phase,
    "enable_thinking": role == "child",
    "bootstrap_leak_level": "solution_replay" if phase == "e0_full_actions" else "action_scaffold",
    "early_rung_bounded": phase == "e0_full_actions",
    "task_bank": {"start_index": int(sys.argv[5]), "count": int(sys.argv[6]), "group_size": 8},
    "reward_mode": "event_control",
    "child_tool_choice_stripped": role == "child",
    "coordinator_tool_choice_stripped": False,
    "env_server_max_concurrent": 1 if role == "coordinator" else 2,
    "promotion_minimum": 4,
    "optimizer_updates": 1,
    "source": {"path": str(source), "model_sha256": digest(source / "model.safetensors")},
    "anchor": {"path": str(anchor), "model_sha256": digest(anchor / "model.safetensors")},
    "output": {"path": str(output), "model_sha256": digest(output / "model.safetensors")},
    "config": {"path": sys.argv[10], "sha256": digest(Path(sys.argv[10]))},
    "bootstrap": {"path": sys.argv[11], "sha256": digest(Path(sys.argv[11]))},
    "metrics": {"path": sys.argv[12], "sha256": digest(Path(sys.argv[12]))},
    "routing_audit": {"path": sys.argv[13], "sha256": digest(Path(sys.argv[13]))},
}
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
attempt_stage=complete
echo "role-GRPO update completed: $receipt"
