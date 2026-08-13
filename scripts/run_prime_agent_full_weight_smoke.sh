#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${MASTERY_GRPO_CONFIG:-$root/configs/debug/subagent-communication/316-qwen35-27b-prime-agent-mastery-grpo.toml}
output=${FULL_WEIGHT_SMOKE_OUTPUT:-/ephemeral/subagent-rung/outputs/317-qwen35-27b-full-weight-smoke}

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -x .venv/bin/rl || ! -x .venv/bin/inference || ! -x .venv/bin/env-server ]]; then
  echo "Prime-RL training executables are missing; run scripts/setup_prime_agent_mastery_host.sh" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "mastery config does not exist: $config" >&2
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

rl @ "$config" \
  --max-steps 1 \
  --output-dir "$output" \
  --deployment.gpus-per-node 8 \
  --deployment.num-train-gpus 6 \
  --deployment.num-infer-gpus 2 \
  --trainer.model.lora None \
  --trainer.ckpt.weights.save-adapter-separately false \
  --orchestrator.eval None \
  --ckpt.interval 1 \
  --ckpt.keep-last 1 \
  --ckpt.keep-interval 1
