#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-delete-spawn-handle-e2e}
output_root=${PRIME_AGENT_E2E_OUTPUT_ROOT:-/ephemeral/subagent-rung/evals/330-prime-agent-delete-spawn-handle-e2e}/${label}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
client_base_url=${EVAL_CLIENT_BASE_URL:-http://127.0.0.1:8401/v1}

: "${PRIME_AGENT_TEST_TARBALL_URL:?set PRIME_AGENT_TEST_TARBALL_URL}"
: "${PRIME_AGENT_TEST_TARBALL_SHA256:?set PRIME_AGENT_TEST_TARBALL_SHA256}"

cd "$root"
"$eval_bin" @ configs/debug/subagent-communication/330-prime-agent-delete-spawn-handle-e2e.toml \
  --model "$model" \
  --output-dir "$output_root" \
  --client.base-url "$client_base_url" \
  --env.agent.harness.tarball-url "$PRIME_AGENT_TEST_TARBALL_URL" \
  --env.agent.harness.tarball-sha256 "$PRIME_AGENT_TEST_TARBALL_SHA256"
