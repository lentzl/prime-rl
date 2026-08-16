#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment="$root/experiments/qwen35-27b-procedural-harness-master-v1"
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-untouched}
output_root=${PROCEDURAL_HARNESS_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-procedural-harness-master-v1}/${label}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
client_base_url=${EVAL_CLIENT_BASE_URL:-}
configs=(valid-baseline ood-baseline)

cd "$root"
uv_bin=${UV_BIN:-$(command -v uv || true)}
if [[ -z "$uv_bin" ]]; then
  echo "uv executable not found" >&2
  exit 1
fi
for package in subagent_communication_v1 procedural_harness_master_v1; do
  "$uv_bin" pip install --python "$root/.venv/bin/python" --no-deps --editable \
    "$root/deps/verifiers/environments/$package" >/dev/null
done
mkdir -p "$output_root"
{
  printf 'prime_rl_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'verifiers_commit=%s\n' "$(git -C deps/verifiers rev-parse HEAD)"
  printf 'prime_agent_version=0.7.2-beta.495.1.97b994c\n'
  printf 'model=%s\n' "$model"
  printf 'model_revision=%s\n' "${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}"
  sha256sum "$experiment"/*.toml
} >"$output_root/VERSIONS.txt"

trace_dirs=()
for name in "${configs[@]}"; do
  args=(
    "$eval_bin" @ "$experiment/$name.toml"
    --model "$model"
    --output-dir "$output_root"
    --run.name "$name"
    --run.dir "$name"
  )
  if [[ -n "$client_base_url" ]]; then
    args+=(--client.base-url "$client_base_url")
  fi
  "${args[@]}"
  trace_dirs+=("$output_root/$name")
  "$root/.venv/bin/python" scripts/summarize_procedural_harness_master_v1.py \
    "$output_root/$name" --output "$output_root/$name/SUMMARY.json" >/dev/null
done

"$root/.venv/bin/python" scripts/summarize_procedural_harness_master_v1.py \
  "${trace_dirs[@]}" --output "$output_root/SUMMARY.json"
