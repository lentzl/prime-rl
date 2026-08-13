#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/ephemeral/subagent-rung/data/programmatic-episodic-memory-v2}"

cd "$ROOT"
uv run python scripts/generate_programmatic_episodic_memory_v2.py \
  --output-dir "$OUT" \
  --seed 20260813 \
  --train-per-family 48 \
  --heldout-per-family 12 \
  --ood-per-family 12

printf '\nMaterialized programmatic episodic memory v2 at %s\n' "$OUT"
printf 'Expected default splits: train=1200 familiar_heldout=300 semantic_ood=96\n'
printf 'Keep familiar_heldout and semantic_ood entirely outside training.\n'
printf 'Prefer train.parquet when datasets/parquet support is available; JSONL is always emitted.\n'
