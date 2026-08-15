#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment="$root/experiments/qwen35-27b-prime-agent-mastery-v2"
model=${1:-Qwen/Qwen3.5-27B}
label=${2:-base}
output_root=${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/evals/qwen35-27b-prime-agent-mastery-v2}/${label}
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
client_base_url=${EVAL_CLIENT_BASE_URL:-}

configs=(
  271-foundations
  272-coordination-calibration
  273-coordination-heldout
  274-ownership-child
  275-ownership-coordinator
  276-oolong
)
if [[ -n "${MASTERY_CONFIGS:-}" ]]; then
  read -ra configs <<<"$MASTERY_CONFIGS"
fi
config_paths=()
for name in "${configs[@]}"; do
  config_paths+=("$experiment/${name}.toml")
done
expected_count=$(
  awk -F' = ' '/^num_tasks = / {sum += $2} END {print sum}' "${config_paths[@]}"
)

cd "$root"
if [[ ! -x "$eval_bin" ]]; then
  echo "eval executable not found: $eval_bin" >&2
  exit 1
fi
mkdir -p "$output_root"
{
  printf 'prime_rl_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'verifiers_commit=%s\n' "$(git -C deps/verifiers rev-parse HEAD)"
  printf 'prime_agent_version=0.7.3\n'
  printf 'model=%s\n' "$model"
  printf 'model_revision=%s\n' "${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}"
  sha256sum "$experiment"/*.toml
} >"$output_root/VERSIONS.txt"

trace_dirs=()
for name in "${configs[@]}"; do
  args=(
    "$eval_bin" @ "$experiment/${name}.toml"
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
done

python scripts/summarize_prime_agent_mastery_v2.py "${trace_dirs[@]}" \
  --expected-count "$expected_count" \
  >"$output_root/SUMMARY.txt"
python scripts/summarize_prime_agent_mastery_v2.py "${trace_dirs[@]}" --json \
  --expected-count "$expected_count" \
  >"$output_root/SUMMARY.json"
