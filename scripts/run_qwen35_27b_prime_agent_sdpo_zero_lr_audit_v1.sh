#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${SDPO_ZERO_LR_CONFIG:-$root/experiments/qwen35-27b-prime-agent-sdpo-v1/zero-lr-audit.toml}

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -x .venv/bin/rl || ! -x .venv/bin/inference || ! -x .venv/bin/env-server ]]; then
  echo "Prime-RL training executables are missing" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "zero-LR SDPO audit config does not exist: $config" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi
.venv/bin/python - "$config" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
if config.get("max_steps") != 1:
    raise SystemExit("zero-LR audit must run exactly one step")
if config.get("trainer", {}).get("optim", {}).get("lr") != 0.0:
    raise SystemExit("zero-LR audit refuses a nonzero learning rate")
if config.get("ckpt", {}).get("interval") is not None:
    raise SystemExit("zero-LR audit must not write a checkpoint")
PY

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN or HF_KEY is required" >&2
  exit 1
fi

if [[ "${SDPO_AUDIT_DRY_RUN:-false}" == true ]]; then
  rl @ "$config" --dry-run
  echo "zero-LR SDPO audit preflight passed"
  exit 0
fi

rl @ "$config"
