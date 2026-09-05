#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revision=${1:?exact materialization revision required}

base_dataset=/home/ubuntu/rlm/artifacts/q35-2b-specialist-population-c1-v1/source-inspector-sft-v1
output_dataset=/home/ubuntu/rlm/artifacts/q35-2b-source-worker-remedial-s5-v1
base_manifest_sha=3eb9ab8816e2fd70be525e926d4b786dde348cfb3bc6473f8415373c9a08b56d
base_parquet_sha=01d7359cac90f5664ff0364520b3c98cc31227c56496b1d78f704712f9a5bab3

cd "$root"
if [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
  echo "materialization revision mismatch" >&2
  exit 1
fi
test "$(sha256sum "$base_dataset/MANIFEST.json" | awk '{print $1}')" = "$base_manifest_sha"
test "$(sha256sum "$base_dataset/train.parquet" | awk '{print $1}')" = "$base_parquet_sha"
if [[ -e "$output_dataset" ]]; then
  echo "refusing to overwrite S5 remedial dataset: $output_dataset" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT=/home/ubuntu/rlm/prime-rl/.venv \
  /home/ubuntu/.local/bin/uv run --no-sync python \
  scripts/export_q35_2b_source_worker_remedial_sft_v1.py \
  --base-dataset-dir "$base_dataset" \
  --output-dir "$output_dataset" \
  --instances-per-variant 8 \
  --instance-offset 60000

echo "S5 remedial manifest SHA-256: $(sha256sum "$output_dataset/MANIFEST.json" | awk '{print $1}')"
echo "S5 remedial parquet SHA-256: $(sha256sum "$output_dataset/train.parquet" | awk '{print $1}')"
echo "Root must independently validate and freeze both hashes before any S5 model call."
