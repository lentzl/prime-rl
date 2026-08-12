#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/prime-rl
output=/ephemeral/subagent-rung/data/217-qwen35-27b-harness-mastery-sft-r1
mkdir -p "$output"

/home/ubuntu/.local/bin/uv run python scripts/export_subagent_communication_sft.py \
  "$output/train.jsonl" \
  --instances 2 \
  --instance-offset 7600 \
  --seed 20270127 \
  --harness-trace /ephemeral/subagent-rung/evals/213-qwen35-27b-clean-causal-smoke/traces.jsonl \
  --families direct single parallel followup \
  --protocol-atoms \
  --binding-focused-atoms \
  --parallel-control-atoms \
  --single-control-atoms \
  --self-contained-child-contract
