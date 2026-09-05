#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revision=${1:?exact execution revision required}
s2_summary_sha=${2:?exact rejected S2 summary SHA-256 required}

e33=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
h176=/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8
s2_candidate=/home/ubuntu/rlm/outputs/q35-2b-specialist-competence-s2-v1/h-source-s2-step8-v1/weights/step_8
s2_summary=/home/ubuntu/rlm/results/q35-2b-specialist-competence-s2-v1/state/s2-source-forced-paired-summary-v1.json
result_root=/home/ubuntu/rlm/results/q35-2b-specialist-competence-s3-low-temperature-v1
state_dir=$result_root/state
config=experiments/qwen35-2b-document-recursion-zero-update-v1/specialist-source-competence-s3-low-temperature-heldout-v1.toml
sampling_contract=experiments/qwen35-2b-document-recursion-zero-update-v1/specialist-source-competence-s3-low-temperature-sampling-contract-v1.json
control_label=s3-source-h176-temperature-zero-forced-control-v1
treatment_label=s3-source-h-source-s2-temperature-zero-forced-treatment-v1
summary=$state_dir/s3-source-temperature-zero-forced-paired-summary-v1.json

e33_sha=e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47
h176_sha=77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e
s2_candidate_sha=937a96154cd47d8dda1fb01125d9f552037a91bc0542748f417652ceddd47f47
config_sha=fff439c348cb5f31946cc549d3ec7688e14c05b6e7cbc930cd0ba7432447a0e7
sampling_sha=da08080c6d1accbbb4ee49c4f6ef891808bf9488942d5d7035faffe0b0c4df98
task_bank_sha=9e029eb5213737aeb63984e601cbdf29911a8d549a3375cc2226a8a261662c66
uv_bin=/home/ubuntu/.local/bin/uv
uv_environment=/home/ubuntu/rlm/prime-rl/.venv

cd "$root"
if [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
  echo "execution revision mismatch" >&2
  exit 1
fi
if [[ "$(git -C deps/verifiers rev-parse --show-toplevel)" != "$root/deps/verifiers" ]]; then
  echo "verifiers submodule is not initialized in this checkout" >&2
  exit 1
fi
test "$(git -C deps/verifiers rev-parse HEAD)" = 5283a85a01b5e8a065b3d2db17f9efa6aa0f3b2f

for value in "$config_sha" "$sampling_sha" "$task_bank_sha" "$s2_summary_sha"; do
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "S3 requires exact prospectively frozen SHA-256 values" >&2
    exit 1
  fi
done

test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha"
test "$(sha256sum "$sampling_contract" | awk '{print $1}')" = "$sampling_sha"
test "$(sha256sum "$s2_summary" | awk '{print $1}')" = "$s2_summary_sha"
test "$(sha256sum "$e33/model.safetensors" | awk '{print $1}')" = "$e33_sha"
test "$(sha256sum "$h176/model.safetensors" | awk '{print $1}')" = "$h176_sha"
test "$(sha256sum "$s2_candidate/model.safetensors" | awk '{print $1}')" = "$s2_candidate_sha"
test -f "$e33/STABLE"
test -f "$h176/STABLE"
test -f "$s2_candidate/STABLE"

UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync python - \
  "$config" "$sampling_contract" <<'PY'
import json
import sys
import tomllib
from pathlib import Path

config = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
contract = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
taskset = config.get("env", {}).get("taskset", {})
sampling = config.get("sampling")
if (
    config.get("num_tasks") != 16
    or config.get("num_rollouts") != 1
    or config.get("shuffle") is not False
    or taskset.get("families")
    != ["specialist_source_ast", "specialist_source_config"]
    or taskset.get("instances_per_template") != 4
    or taskset.get("instance_offset") != 52000
    or taskset.get("seed") != 20270907
    or sampling != {
        "seed": 20270907,
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "reasoning_effort": "high",
        "max_tokens": 4096,
    }
    or contract.get("frozen_before_model_calls") is not True
    or contract.get("environment_sampling")
    != {key: value for key, value in sampling.items() if key != "seed"}
    or contract.get("route_policy", {}).get("mode")
    != "forced_delegate_terminal_source_inspector_assignment"
    or contract.get("route_policy", {}).get("worker_computation_scaffolded")
    is not False
    or contract.get("route_policy", {}).get("worker_parent_send_scaffolded")
    is not False
):
    raise SystemExit("invalid S3 heldout or sampling contract")
PY

UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync python - \
  "$s2_summary" "$h176" "$s2_candidate" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
control_model, treatment_model = sys.argv[2:]
expected_thresholds = {
    "minimum_worker_activations": 16,
    "minimum_treatment_hard_successes": 4,
    "minimum_hard_successes_per_family": 2,
    "minimum_paired_recoveries": 4,
    "maximum_paired_regressions": 0,
    "require_forced_assignment": True,
}
required_true = {
    "fixed_route_exact",
    "router_absent",
    "control_provenance_exact",
    "treatment_provenance_exact",
    "forced_assignment_exact",
}
acceptance = summary.get("acceptance")
inputs = summary.get("inputs")
if (
    summary.get("schema_version")
    != "q35-2b-specialist-competence-paired-summary/v1"
    or summary.get("expert_id") != "source_inspector"
    or summary.get("task_contracts_identical") is not True
    or summary.get("router_taxonomy_evaluated") is not False
    or summary.get("acceptance_gates_relaxed") is not False
    or summary.get("expected_tasks") != 16
    or summary.get("thresholds") != expected_thresholds
    or summary.get("competence_gate_passed") is not False
    or not isinstance(acceptance, dict)
    or any(acceptance.get(key) is not True for key in required_true)
    or not isinstance(inputs, dict)
    or inputs.get("control_model") != control_model
    or inputs.get("treatment_model") != treatment_model
):
    raise SystemExit("S3 requires one immutable, structurally valid, rejected S2 summary")
PY

if [[ -e "$result_root" ]]; then
  echo "refusing duplicate or partial S3 result root: $result_root" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT=/home/ubuntu/rlm/prime-rl/.venv \
  /home/ubuntu/.local/bin/uv run --no-sync python \
  scripts/hash_q35_2b_specialist_task_bank_v1.py "$config" \
  --expected-sha256 "$task_bank_sha"

common_env=(
  PRIME_API_KEY=prime
  UV_PROJECT_ENVIRONMENT=/home/ubuntu/rlm/prime-rl/.venv
  UV_BIN=/home/ubuntu/.local/bin/uv
  INFERENCE_BIN=/home/ubuntu/rlm/prime-rl/.venv/bin/inference
  EVAL_BIN=/home/ubuntu/rlm/prime-rl/.venv/bin/eval
  EVAL_PYTHON_BIN=/home/ubuntu/rlm/prime-rl/.venv/bin/python
  SPECIALIST_COMPETENCE_OUTPUT_ROOT="$result_root"
  SPECIALIST_COMPETENCE_FORCE_FIXED_ACTION=1
)

env "${common_env[@]}" scripts/run_q35_2b_specialist_competence_eval_v1.sh \
  "$e33" "$h176" "$h176" source_inspector "$control_label" "$revision" "$config"

env "${common_env[@]}" scripts/run_q35_2b_specialist_competence_eval_v1.sh \
  "$e33" "$h176" "$s2_candidate" source_inspector "$treatment_label" "$revision" "$config"

UV_PROJECT_ENVIRONMENT=/home/ubuntu/rlm/prime-rl/.venv \
  /home/ubuntu/.local/bin/uv run --no-sync python \
  scripts/summarize_q35_2b_specialist_competence_v1.py \
  --expert-id source_inspector \
  --control-traces "$result_root/$control_label/document/document/traces.jsonl" \
  --control-audit "$result_root/$control_label/ROUTING_AUDIT.jsonl" \
  --treatment-traces "$result_root/$treatment_label/document/document/traces.jsonl" \
  --treatment-audit "$result_root/$treatment_label/ROUTING_AUDIT.jsonl" \
  --control-model "$h176" \
  --treatment-model "$s2_candidate" \
  --expected-tasks 16 \
  --minimum-worker-activations 16 \
  --minimum-treatment-hard-successes 4 \
  --minimum-hard-successes-per-family 2 \
  --minimum-paired-recoveries 4 \
  --maximum-paired-regressions 0 \
  --require-forced-assignment \
  --output "$summary"

test "$(sha256sum "$e33/model.safetensors" | awk '{print $1}')" = "$e33_sha"
test "$(sha256sum "$h176/model.safetensors" | awk '{print $1}')" = "$h176_sha"
test "$(sha256sum "$s2_candidate/model.safetensors" | awk '{print $1}')" = "$s2_candidate_sha"
echo "source worker competence S3 low-temperature screen completed: $summary"
