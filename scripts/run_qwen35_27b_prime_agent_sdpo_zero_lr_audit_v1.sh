#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${SDPO_ZERO_LR_CONFIG:-$root/experiments/qwen35-27b-prime-agent-sdpo-v1/zero-lr-audit.toml}
model_revision=${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}
router_bin=${VLLM_ROUTER_BIN:-$root/.venv/bin/vllm-router}
curl_bin=${CURL_BIN:-curl}

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -x .venv/bin/rl || ! -x .venv/bin/inference || ! -x .venv/bin/env-server ]]; then
  echo "Prime-RL training executables are missing" >&2
  exit 1
fi
if [[ ! -x "$router_bin" ]]; then
  echo "vllm-router is missing; install Prime-RL with its disagg or all extras" >&2
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
prime_agent_version=$(.venv/bin/python - "$config" <<'PY'
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
if not config.get("trainer", {}).get("enable_token_export"):
    raise SystemExit("zero-LR audit must enable token export")
sources = config.get("orchestrator", {}).get("train", {}).get("source", [])
expected = {
    "ownership-child-diagnostic-sdpo": ("sdpo", 1),
    "ownership-coordinator-retention": ("grpo", 2),
    "communication-direct-retention": ("grpo", 2),
    "communication-single-retention": ("grpo", 2),
    "communication-parallel-retention": ("grpo", 2),
    "communication-causal-retention": ("grpo", 2),
}
actual = {
    source.get("name"): (source.get("algo", {}).get("type"), source.get("group_size"))
    for source in sources
}
if actual != expected:
    raise SystemExit(f"zero-LR audit has the wrong mixed source routing: {actual}")
versions = {
    source.get("env", {}).get("agent", {}).get("harness", {}).get("version")
    for source in sources
}
if len(versions) != 1 or None in versions:
    raise SystemExit(f"zero-LR audit must pin one Prime Agent version: {versions}")
print(versions.pop())
PY
)
prime_agent_release_base=https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev
if ! checksums=$("$curl_bin" -fsSL \
  "$prime_agent_release_base/releases/v$prime_agent_version/SHA256SUMS"); then
  echo "Prime Agent artifact is unavailable: $prime_agent_version" >&2
  exit 1
fi
if ! grep -Eq "[[:space:]]prime-agent-$prime_agent_version\\.tgz$" <<<"$checksums"; then
  echo "Prime Agent checksum manifest lacks its package: $prime_agent_version" >&2
  exit 1
fi

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
model_snapshot=$(
  .venv/bin/python - "$model_revision" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download("Qwen/Qwen3.5-27B", revision=sys.argv[1]))
PY
)
if [[ "$(basename "$model_snapshot")" != "$model_revision" ]]; then
  echo "resolved model snapshot does not match the pinned revision" >&2
  exit 1
fi

if [[ "${SDPO_AUDIT_DRY_RUN:-false}" == true ]]; then
  rl @ "$config" --model.name "$model_snapshot" --dry-run
  echo "zero-LR SDPO audit preflight passed"
  exit 0
fi

rl @ "$config" --model.name "$model_snapshot"

run_dir=/ephemeral/outputs/qwen35-27b-prime-agent-sdpo-v1/zero-lr-audit
.venv/bin/python scripts/validate_prime_agent_sdpo_zero_lr_audit_v1.py \
  "$run_dir" \
  --expected-revision "$model_revision" \
  --output "$run_dir/AUDIT.json" | tee "$run_dir/AUDIT.txt"
