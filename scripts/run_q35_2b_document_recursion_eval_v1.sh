#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:?model name required}
label=${2:?evaluation label required}
config=${DOCUMENT_RECURSION_CONFIG:?DOCUMENT_RECURSION_CONFIG is required}
output_root=${QWEN38_QUALIFICATION_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-document-recursion-zero-update-v1}
run_output=$output_root/$label/document
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
runtime_python=${EVAL_PYTHON_BIN:-$root/.venv/bin/python}
uv_bin=${UV_BIN:-$(command -v uv || true)}

if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin=$HOME/.local/bin/uv
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv executable not found" >&2
  exit 1
fi
if [[ ! -x "$eval_bin" ]]; then
  echo "eval executable not found: $eval_bin" >&2
  exit 1
fi
if [[ ! -x "$runtime_python" ]]; then
  echo "evaluation Python is missing: $runtime_python" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "document recursion config not found: $config" >&2
  exit 1
fi
expected_count=$("$runtime_python" - "$config" <<'PY'
import sys
import tomllib
from pathlib import Path

config = tomllib.loads(Path(sys.argv[1]).read_text())
num_tasks = config.get("num_tasks")
num_rollouts = config.get("num_rollouts")
if not isinstance(num_tasks, int) or num_tasks < 1:
    raise SystemExit("document recursion config requires a positive num_tasks")
if not isinstance(num_rollouts, int) or num_rollouts < 1:
    raise SystemExit("document recursion config requires a positive num_rollouts")
print(num_tasks * num_rollouts)
PY
)

cd "$root"
mkdir -p "$run_output"
{
  printf 'prime_rl_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'verifiers_commit=%s\n' "$(git -C deps/verifiers rev-parse HEAD)"
  printf 'model=%s\n' "$model"
  printf 'model_revision=%s\n' "${MODEL_REVISION:-candidate-local}"
  sha256sum "$config"
} >"$run_output/VERSIONS.txt"

"$eval_bin" @ "$config" \
  --model "$model" \
  --client.base-url "${EVAL_CLIENT_BASE_URL:?EVAL_CLIENT_BASE_URL is required}" \
  --output-dir "$run_output" \
  --run.name document \
  --run.dir document

"$uv_bin" run --no-sync scripts/summarize_prime_agent_mastery_v2.py \
  "$run_output/document" \
  --expected-count "$expected_count" \
  >"$run_output/SUMMARY.txt"
"$uv_bin" run --no-sync scripts/summarize_prime_agent_mastery_v2.py \
  "$run_output/document" \
  --expected-count "$expected_count" \
  --json \
  >"$run_output/SUMMARY.json"
