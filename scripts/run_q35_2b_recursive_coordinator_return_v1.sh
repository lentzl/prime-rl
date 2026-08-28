#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
label=${2:?evaluation label required}
revision=${3:?model revision required}
bootstrap_path=${4:?train-gen bootstrap artifact required}

if [[ ! -f "$bootstrap_path" ]]; then
  echo "bootstrap artifact does not exist: $bootstrap_path" >&2
  exit 1
fi

export QWEN38_QUALIFICATION_CONFIG="$root/experiments/qwen35-2b-recursive-coordinator-return-v1/qualification-template.toml"
export QWEN38_QUALIFICATION_OUTPUT_ROOT="${C_RETURN_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-recursive-coordinator-return-v1}"
export QWEN38_QUALIFICATION_AXES=natural_n1a
export QWEN38_QUALIFICATION_NUM_TASKS="${C_RETURN_NUM_TASKS:-6}"
export QWEN38_QUALIFICATION_NUM_ROLLOUTS=1
export QWEN38_QUALIFICATION_MAX_CONCURRENT=1
export QWEN38_QUALIFICATION_START_INDEX="${C_RETURN_START_INDEX:-9420000}"
export QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH="$bootstrap_path"
export PROCEDURAL_INTERACTION_CURRICULUM=e0c4_recursive_coordinator_return
export DUAL_EXTERNAL_MODEL=q35-2b-recursive-coordinator-return
export UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
export PATH="$root/.venv/bin:$PATH"

exec "$root/scripts/run_q35_2b_dual_policy_mastery_v1.sh" \
  "$coordinator_model" "$coordinator_model" "$label" "$revision"
