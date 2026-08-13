#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${MASTERY_GRPO_CONFIG:-$root/configs/debug/subagent-communication/316-qwen35-27b-prime-agent-mastery-grpo.toml}
output=${FULL_WEIGHT_BALANCED_OUTPUT:-/ephemeral/subagent-rung/outputs/327-qwen35-27b-full-weight-balanced-r1}
max_steps=${FULL_WEIGHT_MAX_STEPS:-2}

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
if [[ -e "$output" ]]; then
  echo "refusing to overwrite balanced-mastery output: $output" >&2
  exit 1
fi
if (( max_steps < 1 )); then
  echo "FULL_WEIGHT_MAX_STEPS must be positive" >&2
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

# Keep the source config's intended six GRPO groups per update. The earlier
# qualification launcher deliberately reduced this to one group, which is not
# representative of the broad mastery objective.
rl @ "$config" \
  --max-steps "$max_steps" \
  --output-dir "$output" \
  --deployment.gpus-per-node 8 \
  --deployment.num-train-gpus 6 \
  --deployment.num-infer-gpus 2 \
  --inference.vllm.tensor-parallel-size 2 \
  --inference.vllm.gpu-memory-utilization 0.80 \
  --inference.vllm.max-num-seqs 4 \
  --inference.vllm.enforce-eager true \
  --trainer.model.lora None \
  --trainer.ckpt.weights.save-adapter-separately false \
  --orchestrator.eval None \
  --orchestrator.batch-size 24 \
  --orchestrator.oversampling-factor 1.0 \
  --ckpt.interval 2 \
  --ckpt.keep-last 1 \
  --ckpt.keep-interval 2
