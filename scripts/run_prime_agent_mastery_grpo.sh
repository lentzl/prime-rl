#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${MASTERY_GRPO_CONFIG:-$root/configs/debug/subagent-communication/316-qwen35-27b-prime-agent-mastery-grpo.toml}
backup_repo=${MASTERY_BACKUP_REPO:-lentzl/rlm-prime-agent-training-backups}
run_id=${MASTERY_RUN_ID:-316-qwen35-27b-prime-agent-mastery-grpo-r64}

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

read -r output max_step < <(
  .venv/bin/python - "$config" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
print(config["output_dir"], config["max_steps"])
PY
)
backup_log=${MASTERY_BACKUP_LOG:-/ephemeral/subagent-rung/logs/$run_id-hf-adapter-backup.log}
mkdir -p "$(dirname "$backup_log")"
.venv/bin/python scripts/backup_prime_agent_adapters.py \
  "$output" "$backup_repo" "$run_id" --max-step "$max_step" --training-config "$config" \
  >"$backup_log" 2>&1 &
backup_pid=$!
cleanup() {
  kill "$backup_pid" 2>/dev/null || true
  wait "$backup_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

set +e
rl @ "$config"
status=$?
set -e
cleanup
trap - EXIT INT TERM
.venv/bin/python scripts/backup_prime_agent_adapters.py \
  "$output" "$backup_repo" "$run_id" --once --training-config "$config" \
  >>"$backup_log" 2>&1 || true
exit "$status"
