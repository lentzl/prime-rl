#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/prime-rl
output=/ephemeral/subagent-rung/data/221-qwen35-27b-bidirectional-sft-r1
mkdir -p "$output"

/home/ubuntu/.local/bin/uv run python scripts/export_subagent_communication_sft.py \
  "$output/train.jsonl" \
  --instances 2 \
  --instance-offset 8000 \
  --seed 20270131 \
  --harness-trace /ephemeral/subagent-rung/evals/213-qwen35-27b-clean-causal-smoke/traces.jsonl \
  --families direct single parallel followup handshake \
  --followup-copies 4 \
  --handshake-copies 4 \
  --self-contained-child-contract

sha256sum "$output/train.jsonl"
