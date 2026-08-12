#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/prime-rl
for config in 214-qwen35-27b-standard-dev 215-qwen35-27b-bidirectional-dev 216-qwen35-27b-clean-causal-variance; do
  /home/ubuntu/.local/bin/uv run eval \
    @ "configs/debug/subagent-communication/${config}.toml"
done
