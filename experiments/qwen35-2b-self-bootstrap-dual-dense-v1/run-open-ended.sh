#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
data_root=${Q35_DUAL_DENSE_DATA_ROOT:-/home/ubuntu/rlm}
experiment_dir=$root/experiments/qwen35-2b-self-bootstrap-dual-dense-v1

cd "$root"
exec /home/ubuntu/.local/bin/uv run --frozen --no-sync \
  scripts/run_q35_2b_spade_dual_dense_autonomous_v1.py \
  --repo-root "$root" \
  --events "$experiment_dir/events.jsonl" \
  --artifacts-root "$data_root/artifacts/q35-2b-self-bootstrap-dual-dense-v1" \
  --results-root "$data_root/results/q35-2b-self-bootstrap-dual-dense-v1" \
  --output-root "$data_root/outputs/q35-2b-self-bootstrap-dual-dense-v1" \
  --journal "$experiment_dir/journal.jsonl" \
  --stop-file "$experiment_dir/STOP" \
  --lock-file "$experiment_dir/runner.lock" \
  --experiment-dir "$experiment_dir" \
  --learning-rate 1e-6 \
  --coevolution \
  --open-ended
