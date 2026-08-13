#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/ephemeral/subagent-rung/data/programmatic-episodic-memory-v1}"

cd "$ROOT"
uv run python scripts/generate_programmatic_episodic_memory_v1.py \
  --output-dir "$OUT" \
  --seed 20260813 \
  --train-per-family 6 \
  --heldout-per-family 3 \
  --ood-per-family 6

printf '\nMaterialized programmatic episodic memory dataset at %s\n' "$OUT"
printf 'Expected default splits: train=72 familiar_heldout=36 semantic_ood=18\n'
printf 'Use train.parquet for bootstrap training when datasets/parquet support is available.\n'
printf 'Keep familiar_heldout and semantic_ood entirely outside training.\n'
