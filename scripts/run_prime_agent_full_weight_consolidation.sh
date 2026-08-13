#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_config=${MASTERY_GRPO_CONFIG:-$root/configs/debug/subagent-communication/316-qwen35-27b-prime-agent-mastery-grpo.toml}
output=${FULL_WEIGHT_CONSOLIDATION_OUTPUT:-/ephemeral/subagent-rung/outputs/331-qwen35-27b-full-weight-consolidation-r1}
resolved_config=${FULL_WEIGHT_CONSOLIDATION_CONFIG:-/ephemeral/subagent-rung/configs/331-qwen35-27b-full-weight-consolidation-r1.toml}

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -x .venv/bin/rl || ! -x .venv/bin/inference || ! -x .venv/bin/env-server ]]; then
  echo "Prime-RL training executables are missing; run scripts/setup_prime_agent_mastery_host.sh" >&2
  exit 1
fi
if [[ ! -f "$source_config" ]]; then
  echo "mastery source config does not exist: $source_config" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite consolidation output: $output" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

set -a
source .env
set +a
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN or HF_KEY is required" >&2
  exit 1
fi

.venv/bin/python scripts/prepare_prime_agent_full_weight_consolidation.py \
  "$source_config" \
  "$resolved_config" \
  "$output"
rl @ "$resolved_config"
