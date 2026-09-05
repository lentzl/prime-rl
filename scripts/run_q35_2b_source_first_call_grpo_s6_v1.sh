#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revision=${1:?exact execution revision required}
verifiers_revision=${2:?exact durable Verifiers revision required}
audit_config_sha=${3:?root-frozen audit config SHA-256 required}
update_config_sha=${4:?root-frozen update config SHA-256 required}
heldout_config=${5:?root-materialized heldout config required}
heldout_config_sha=${6:?root-frozen heldout config SHA-256 required}
sampling_contract=${7:?root-materialized sampling contract required}
sampling_contract_sha=${8:?root-frozen sampling contract SHA-256 required}
task_bank_sha=${9:?root-frozen heldout task-bank SHA-256 required}
s5_summary_sha=${10:?exact rejected S5 primary-summary SHA-256 required}
dry_run=${S6_DRY_RUN:-false}

audit_config=experiments/qwen35-2b-document-recursion-zero-update-v1/\
specialist-source-competence-s6-first-call-grpo-zero-lr.toml
update_config=experiments/qwen35-2b-document-recursion-zero-update-v1/\
specialist-source-competence-s6-first-call-grpo-step1.toml
validator=scripts/validate_q35_2b_source_first_call_grpo_s6_v1.py

e33=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/\
c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
h176=/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/\
h176child8-document-child-real12-step8-v2/weights/step_8
s2=/home/ubuntu/rlm/outputs/q35-2b-specialist-competence-s2-v1/\
h-source-s2-step8-v1/weights/step_8
s5=/home/ubuntu/rlm/outputs/q35-2b-source-worker-remedial-s5-v1/\
h-source-s5-remedial-step8-v1/weights/step_8
s5_summary=/home/ubuntu/rlm/results/q35-2b-source-worker-remedial-s5-v1/state/\
s5-source-h176-versus-remedial-primary-summary-v1.json

output_root=/home/ubuntu/rlm/outputs/q35-2b-source-first-call-s6-v1
result_root=/home/ubuntu/rlm/results/q35-2b-source-first-call-s6-v1
audit_output=$output_root/zero-lr-audit
update_output=$output_root/step1
candidate=$update_output/weights/step_1
audit_result=$result_root/source-first-call-s6-zero-lr-audit.json
update_result=$result_root/source-first-call-s6-step1-validation.json
receipt=$result_root/source-first-call-s6-step1-receipt.json

e33_sha=e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47
h176_sha=77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e
s2_sha=937a96154cd47d8dda1fb01125d9f552037a91bc0542748f417652ceddd47f47
s5_sha=09a2e3e88030d17896e554211d8fc7eff709d6b4a619e99e3342d05aacde0782
expected_s5_summary_sha=cda2cdba4b9e6e8519f3008b4a49bf178474b8d4e6c42384e634dae58b45f90e
legacy_verifiers_gitlink=5283a85a01b5e8a065b3d2db17f9efa6aa0f3b2f
live_detached_verifiers_prefix=53bafca
uv_bin=/home/ubuntu/.local/bin/uv
uv_environment=/home/ubuntu/rlm/prime-rl/.venv

cd "$root"
if [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
  echo "S6 execution revision mismatch" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "S6 requires a clean parent worktree" >&2
  exit 1
fi
if [[ "$(git -C deps/verifiers rev-parse --show-toplevel 2>/dev/null)" != "$root/deps/verifiers" ]]; then
  echo "S6 requires an initialized Verifiers submodule" >&2
  exit 1
fi
actual_verifiers_revision=$(git -C deps/verifiers rev-parse HEAD)
if [[ "$actual_verifiers_revision" != "$verifiers_revision" ]]; then
  echo "S6 Verifiers revision mismatch" >&2
  exit 1
fi
if [[ "$actual_verifiers_revision" == "$legacy_verifiers_gitlink" || "$actual_verifiers_revision" == "$live_detached_verifiers_prefix"* ]]; then
  echo "S6 refuses the legacy gitlink and detached live Verifiers revisions" >&2
  exit 1
fi
if [[ -n "$(git -C deps/verifiers status --porcelain)" ]]; then
  echo "S6 requires a clean durable Verifiers checkout" >&2
  exit 1
fi
for package in subagent_communication_v1 source_worker_first_call_v1; do
  if [[ ! -f "$root/deps/verifiers/environments/$package/pyproject.toml" ]]; then
    echo "S6 Verifiers package is missing: $package" >&2
    exit 1
  fi
done

for value in \
  "$audit_config_sha" \
  "$update_config_sha" \
  "$heldout_config_sha" \
  "$sampling_contract_sha" \
  "$task_bank_sha" \
  "$s5_summary_sha"; do
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "S6 requires exact root-frozen SHA-256 values" >&2
    exit 1
  fi
done
if [[ "$s5_summary_sha" != "$expected_s5_summary_sha" ]]; then
  echo "S6 rejected S5 summary identity mismatch" >&2
  exit 1
fi
test "$(sha256sum "$audit_config" | awk '{print $1}')" = "$audit_config_sha"
test "$(sha256sum "$update_config" | awk '{print $1}')" = "$update_config_sha"
test "$(sha256sum "$heldout_config" | awk '{print $1}')" = "$heldout_config_sha"
test "$(sha256sum "$sampling_contract" | awk '{print $1}')" = "$sampling_contract_sha"
test "$(sha256sum "$s5_summary" | awk '{print $1}')" = "$s5_summary_sha"

for model_and_sha in \
  "$e33:$e33_sha" \
  "$h176:$h176_sha" \
  "$s2:$s2_sha" \
  "$s5:$s5_sha"; do
  model=${model_and_sha%:*}
  expected_sha=${model_and_sha##*:}
  test -f "$model/STABLE"
  test "$(sha256sum "$model/model.safetensors" | awk '{print $1}')" = "$expected_sha"
done

"$uv_bin" run --frozen --no-sync python "$validator" "$audit_config" --stage audit
"$uv_bin" run --frozen --no-sync python "$validator" "$update_config" --stage update
UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync python - \
  "$s5_summary" "$heldout_config" "$sampling_contract" <<'PY'
import json
import sys
import tomllib
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
config = tomllib.loads(Path(sys.argv[2]).read_text())
contract = json.loads(Path(sys.argv[3]).read_text())
expected_thresholds = {
    "minimum_worker_activations": 16,
    "minimum_treatment_hard_successes": 4,
    "minimum_hard_successes_per_family": 2,
    "minimum_paired_recoveries": 4,
    "maximum_paired_regressions": 0,
    "require_forced_assignment": True,
}
if (
    summary.get("schema_version") != "q35-2b-specialist-competence-paired-summary/v1"
    or summary.get("competence_gate_passed") is not False
    or summary.get("acceptance_gates_relaxed") is not False
    or summary.get("thresholds") != expected_thresholds
    or summary.get("control", {}).get("hard_successes") != 0
    or summary.get("treatment", {}).get("hard_successes") != 0
    or summary.get("treatment", {}).get("exact_answers") != 0
):
    raise SystemExit("S6 requires the exact structurally rejected S5 primary summary")
taskset = config.get("env", {}).get("taskset", {})
sampling = config.get("sampling", {})
if (
    config.get("num_tasks") != 16
    or config.get("num_rollouts") != 1
    or config.get("shuffle") is not False
    or taskset.get("split") != "eval"
    or taskset.get("families") != ["specialist_source_ast", "specialist_source_config"]
    or taskset.get("instances_per_template") != 4
    or not isinstance(taskset.get("seed"), int)
    or not isinstance(taskset.get("instance_offset"), int)
    or taskset.get("seed") == 20270909
    or taskset.get("instance_offset") == 70000
    or sampling.get("temperature") != 0.0
    or sampling.get("seed") != taskset.get("seed")
    or contract.get("frozen_before_model_calls") is not True
    or contract.get("admission", {}).get("acceptance_gates_relaxed") is not False
    or contract.get("admission", {}).get("minimum_treatment_hard_successes") != 4
    or contract.get("admission", {}).get("minimum_hard_successes_per_family") != 2
    or contract.get("admission", {}).get("minimum_paired_recoveries") != 4
    or contract.get("admission", {}).get("maximum_paired_regressions") != 0
):
    raise SystemExit("S6 heldout or sampling contract is not prospectively frozen")
PY

if [[ -e "$output_root" || -e "$result_root" ]]; then
  echo "refusing duplicate or partial S6 output/result root" >&2
  exit 1
fi
UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync \
  --with-editable "$root/deps/verifiers/environments/subagent_communication_v1" \
  python \
  scripts/hash_q35_2b_specialist_task_bank_v1.py "$heldout_config" \
  --expected-sha256 "$task_bank_sha"

if [[ "$dry_run" != true ]]; then
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  if [[ ${#gpu_names[@]} -ne 2 ]] || printf '%s\n' "${gpu_names[@]}" | grep -qv 'RTX A6000'; then
    echo "S6 requires exactly 2x RTX A6000" >&2
    exit 1
  fi
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "refusing S6 launch while another GPU process is active" >&2
    exit 1
  fi
fi

export PATH="$uv_environment/bin:$HOME/.local/bin:$PATH"
run_s6_rl() {
  UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync \
    --with-editable "$root/deps/verifiers/environments/subagent_communication_v1" \
    --with-editable "$root/deps/verifiers/environments/source_worker_first_call_v1" \
    rl @ "$@"
}
run_s6_python() {
  UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync \
    --with-editable "$root/deps/verifiers/environments/subagent_communication_v1" \
    --with-editable "$root/deps/verifiers/environments/source_worker_first_call_v1" \
    python "$@"
}
if [[ "$dry_run" == true ]]; then
  run_s6_rl "$audit_config" --dry-run
  run_s6_rl "$update_config" --dry-run
  echo "S6 static and launcher dry-run checks passed"
  exit 0
fi
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

run_s6_rl "$audit_config"
mkdir -p "$result_root"
run_s6_python "$validator" "$audit_output" \
  --runtime --output "$audit_result"
for model_and_sha in \
  "$e33:$e33_sha" \
  "$h176:$h176_sha" \
  "$s2:$s2_sha" \
  "$s5:$s5_sha"; do
  model=${model_and_sha%:*}
  expected_sha=${model_and_sha##*:}
  test "$(sha256sum "$model/model.safetensors" | awk '{print $1}')" = "$expected_sha"
done

run_s6_rl "$update_config"
test -f "$candidate/STABLE"
test -f "$candidate/model.safetensors"
run_s6_python "$validator" "$update_output" --runtime --stage update \
  --output "$update_result"
candidate_sha=$(sha256sum "$candidate/model.safetensors" | awk '{print $1}')
if [[ "$candidate_sha" == "$s5_sha" ]]; then
  echo "S6 update did not change model weights" >&2
  exit 1
fi
UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync python - \
  "$update_output/metrics.jsonl" <<'PY'
import json
import math
import sys
from pathlib import Path

records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
def last(key):
    values = [record[key] for record in records if key in record]
    if not values or not isinstance(values[-1], (int, float)) or not math.isfinite(values[-1]):
        raise SystemExit(f"missing finite S6 update metric: {key}")
    return float(values[-1])
if last("optim/lr") != 1e-6 or last("optim/update_succeeded") != 1 or last("optim/grad_norm") <= 0:
    raise SystemExit("S6 conditional update did not complete with its predeclared gradient")
PY

UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync python - \
  "$receipt" "$revision" "$verifiers_revision" "$audit_result" "$candidate" \
  "$candidate_sha" "$heldout_config" "$sampling_contract" "$task_bank_sha" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

destination = Path(sys.argv[1])
payload = {
    "schema_version": "q35-2b-source-first-call-grpo-update/v1",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "execution_revision": sys.argv[2],
    "verifiers_revision": sys.argv[3],
    "zero_lr_audit": {"path": sys.argv[4], "sha256": digest(sys.argv[4])},
    "source": {"label": "S5", "model_sha256": "09a2e3e88030d17896e554211d8fc7eff709d6b4a619e99e3342d05aacde0782"},
    "candidate": {"path": sys.argv[5], "model_sha256": sys.argv[6]},
    "optimizer_updates": 1,
    "update_type": "full_dense",
    "role": "non_root_source_inspector",
    "heldout_config": {"path": sys.argv[7], "sha256": digest(sys.argv[7])},
    "sampling_contract": {"path": sys.argv[8], "sha256": digest(sys.argv[8])},
    "heldout_task_bank_sha256": sys.argv[9],
    "admitted": False,
}
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

common_env=(
  PRIME_API_KEY=prime
  UV_PROJECT_ENVIRONMENT="$uv_environment"
  UV_BIN="$uv_bin"
  INFERENCE_BIN="$uv_environment/bin/inference"
  EVAL_BIN="$uv_environment/bin/eval"
  EVAL_PYTHON_BIN="$uv_environment/bin/python"
  SPECIALIST_COMPETENCE_OUTPUT_ROOT="$result_root"
  SPECIALIST_COMPETENCE_FORCE_FIXED_ACTION=1
)
control_label=s6-h176-control-v1
s2_label=s6-s2-bridge-v1
s5_label=s6-s5-bridge-v1
treatment_label=s6-treatment-v1
env "${common_env[@]}" scripts/run_q35_2b_specialist_competence_eval_v1.sh \
  "$e33" "$h176" "$h176" source_inspector "$control_label" "$revision" "$heldout_config"
env "${common_env[@]}" scripts/run_q35_2b_specialist_competence_eval_v1.sh \
  "$e33" "$h176" "$s2" source_inspector "$s2_label" "$revision" "$heldout_config"
env "${common_env[@]}" scripts/run_q35_2b_specialist_competence_eval_v1.sh \
  "$e33" "$h176" "$s5" source_inspector "$s5_label" "$revision" "$heldout_config"
env "${common_env[@]}" scripts/run_q35_2b_specialist_competence_eval_v1.sh \
  "$e33" "$h176" "$candidate" source_inspector "$treatment_label" "$revision" "$heldout_config"

summarizer=(
  UV_PROJECT_ENVIRONMENT="$uv_environment"
  "$uv_bin" run --no-sync python
  scripts/summarize_q35_2b_specialist_competence_v1.py
  --expert-id source_inspector
  --expected-tasks 16
  --minimum-worker-activations 16
  --minimum-treatment-hard-successes 4
  --minimum-hard-successes-per-family 2
  --minimum-paired-recoveries 4
  --maximum-paired-regressions 0
  --require-forced-assignment
)
traces() { printf '%s/%s/document/document/traces.jsonl' "$result_root" "$1"; }
audit() { printf '%s/%s/ROUTING_AUDIT.jsonl' "$result_root" "$1"; }
summarize_pair() {
  local control_label=$1 control_model=$2 treatment_label=$3 treatment_model=$4 output=$5
  env "${summarizer[@]}" \
    --control-traces "$(traces "$control_label")" \
    --control-audit "$(audit "$control_label")" \
    --treatment-traces "$(traces "$treatment_label")" \
    --treatment-audit "$(audit "$treatment_label")" \
    --control-model "$control_model" \
    --treatment-model "$treatment_model" \
    --output "$output"
}
summarize_pair "$s2_label" "$s2" "$treatment_label" "$candidate" \
  "$result_root/s6-s2-versus-treatment.json"
summarize_pair "$s5_label" "$s5" "$treatment_label" "$candidate" \
  "$result_root/s6-s5-versus-treatment.json"
summarize_pair "$control_label" "$h176" "$treatment_label" "$candidate" \
  "$result_root/s6-h176-versus-treatment-primary.json"

for model_and_sha in \
  "$e33:$e33_sha" \
  "$h176:$h176_sha" \
  "$s2:$s2_sha" \
  "$s5:$s5_sha"; do
  model=${model_and_sha%:*}
  expected_sha=${model_and_sha##*:}
  test "$(sha256sum "$model/model.safetensors" | awk '{print $1}')" = "$expected_sha"
done
echo "S6 completed: $result_root/s6-h176-versus-treatment-primary.json"
