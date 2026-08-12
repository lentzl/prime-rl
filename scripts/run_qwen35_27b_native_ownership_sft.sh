#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/prime-rl
SOURCE1=/ephemeral/subagent-rung/evals/249-qwen35-27b-guided-native-ownership-collection-r1/traces.jsonl
SOURCE2=/ephemeral/subagent-rung/evals/250-qwen35-27b-guided-native-ownership-collection-r2/traces.jsonl
DATA=/ephemeral/subagent-rung/data/251-qwen35-27b-native-ownership-sft-r1
OUTPUT=/ephemeral/subagent-rung/outputs/251-qwen35-27b-native-ownership-sft-r1
CONFIG=configs/debug/subagent-communication/251-qwen35-27b-native-ownership-sft.toml

cd "$ROOT"
rm -rf "$DATA" "$OUTPUT"
export HF_TOKEN="$HF_KEY"

.venv/bin/python scripts/export_ownership_teacher_sft.py \
  "$SOURCE1" "$SOURCE2" \
  --output-dir "$DATA" \
  --min-traces 6 \
  --max-per-task 1

.venv/bin/python -c \
  'from datasets import load_dataset; data = load_dataset("json", data_files="'$DATA'/train.json", split="train"); assert len(data) >= 6'

systemctl --user stop qwen35-27b-teacher-inference.service
exec .venv/bin/sft @ "$CONFIG"
