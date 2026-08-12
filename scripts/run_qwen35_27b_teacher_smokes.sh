#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/prime-rl
/home/ubuntu/.local/bin/uv run eval \
  @ configs/debug/subagent-communication/212-qwen35-27b-direct-smoke.toml
/home/ubuntu/.local/bin/uv run eval \
  @ configs/debug/subagent-communication/213-qwen35-27b-clean-causal-smoke.toml
