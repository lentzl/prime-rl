#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
DATA=/ephemeral/subagent-rung/data/243-qwen35-27b-thinking-ownership-bootstrap-sft-r1
OUTPUT=/ephemeral/subagent-rung/outputs/243-qwen35-27b-thinking-ownership-bootstrap-sft-r1
CONFIG=configs/debug/subagent-communication/243-qwen35-27b-thinking-ownership-bootstrap-sft.toml

cd "$ROOT"
rm -rf "$DATA" "$OUTPUT"
mkdir -p "$DATA"

/home/ubuntu/prime-rl/.venv/bin/python scripts/export_subagent_communication_sft.py \
  "$DATA/train.jsonl" \
  --instances 8 \
  --instance-offset 10000 \
  --seed 20270221 \
  --families followup \
  --retained-spawn-only \
  --thinking-rationales

exec /home/ubuntu/prime-rl/.venv/bin/sft @ "$CONFIG"
