#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:?model or served model name is required}
label=${2:?run label is required}
output_root=${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/evals/qwen35-2b-self-bootstrap-v1}
run_output=$output_root/$label
runtime_python=${EVAL_PYTHON_BIN:-$root/.venv/bin/python}
base_url=${EVAL_CLIENT_BASE_URL:-http://127.0.0.1:8100/v1}
model_revision=${MODEL_REVISION:?MODEL_REVISION is required}
axes_csv=${SPADE_RUNG0_AXES:-natural_direct_control,natural_n1a,natural_n1a_local,natural_n1b}
tasks_per_axis=${SPADE_RUNG0_TASKS_PER_AXIS:-8}
no_hint_seeds_csv=${SPADE_RUNG0_NO_HINT_SEEDS:-20260822,20260823,20260824,20260825}
hinted_seeds_csv=${SPADE_RUNG0_HINTED_SEEDS:-20260826,20260827,20260828,20260829}
temperature=${SPADE_RUNG0_TEMPERATURE:-0.6}
reasoning_effort=${SPADE_RUNG0_REASONING_EFFORT:-high}
max_concurrent=${SPADE_RUNG0_MAX_CONCURRENT:-4}
hint_seed=${SPADE_RUNG0_HINT_SEED:-20260830}
no_hint_source_root=${SPADE_RUNG0_NO_HINT_SOURCE_ROOT:-}
hint_artifact=$run_output/privileged-hints.json
summary=$run_output/RUNG0_SUMMARY.json

cd "$root"
uv_bin=${UV_BIN:-$(command -v uv || true)}
if [[ -z "$uv_bin" || ! -x "$runtime_python" ]]; then
  echo "uv and the evaluation Python are required" >&2
  exit 1
fi
if [[ ! "$tasks_per_axis" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPADE_RUNG0_TASKS_PER_AXIS must be a positive integer" >&2
  exit 1
fi
if [[ -e "$hint_artifact" || -e "$summary" || -e "$run_output/RUNG0_VERSIONS.txt" ]]; then
  echo "refusing to overwrite existing Rung 0 artifacts in $run_output" >&2
  exit 1
fi

IFS=, read -ra axes <<<"$axes_csv"
IFS=, read -ra no_hint_seeds <<<"$no_hint_seeds_csv"
IFS=, read -ra hinted_seeds <<<"$hinted_seeds_csv"
if [[ ${#no_hint_seeds[@]} -ne ${#hinted_seeds[@]} ]]; then
  echo "Rung 0 arms must have equal rollout counts" >&2
  exit 1
fi
declare -A seen_seeds=()
for seed in "${no_hint_seeds[@]}" "${hinted_seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ || -n "${seen_seeds[$seed]:-}" ]]; then
    echo "Rung 0 seeds must be unique non-negative integers: $seed" >&2
    exit 1
  fi
  seen_seeds[$seed]=1
done

hint_axis_args=()
for axis in "${axes[@]}"; do
  case "$axis" in
    natural_direct_control) start_index=4007000 ;;
    natural_n1a) start_index=4004000 ;;
    natural_n1a_local) start_index=4006000 ;;
    natural_n1b) start_index=4005000 ;;
    *) echo "unsupported Rung 0 axis: $axis" >&2; exit 1 ;;
  esac
  hint_axis_args+=(--axis "$axis:$start_index")
done

for package in subagent_communication_v1 procedural_harness_master_v1; do
  "$uv_bin" pip install --python "$runtime_python" --no-deps --editable \
    "$root/deps/verifiers/environments/$package" >/dev/null
done

"$runtime_python" scripts/build_q35_2b_spade_rung0_hints_v1.py \
  --base-url "$base_url" \
  --model "$model" \
  --model-revision "$model_revision" \
  --output "$hint_artifact" \
  --tasks-per-axis "$tasks_per_axis" \
  --sampling-seed "$hint_seed" \
  --temperature "$temperature" \
  "${hint_axis_args[@]}"

{
  printf 'schema_version=qwen35-2b-spade-rung0-versions/v1\n'
  printf 'gradient_updates=0\n'
  printf 'model=%s\n' "$model"
  printf 'model_revision=%s\n' "$model_revision"
  printf 'axes=%s\n' "$axes_csv"
  printf 'tasks_per_axis=%s\n' "$tasks_per_axis"
  printf 'no_hint_seeds=%s\n' "$no_hint_seeds_csv"
  printf 'hinted_seeds=%s\n' "$hinted_seeds_csv"
  printf 'temperature=%s\n' "$temperature"
  printf 'reasoning_effort=%s\n' "$reasoning_effort"
  printf 'max_concurrent=%s\n' "$max_concurrent"
  printf 'hint_seed=%s\n' "$hint_seed"
  printf 'no_hint_source_root=%s\n' "${no_hint_source_root:-none}"
  sha256sum \
    "$hint_artifact" \
    "$root/experiments/qwen35-2b-self-bootstrap-v1/preregistration.json" \
    "$root/experiments/qwen35-2b-self-bootstrap-v1/paper-alignment-addendum.json" \
    "$root/experiments/qwen35-2b-self-bootstrap-v1/direct-control-policy-amendment.json" \
    "$root/scripts/build_q35_2b_spade_rung0_hints_v1.py" \
    "$root/scripts/summarize_q35_2b_spade_rung0_v1.py" \
    "$root/scripts/run_q35_2b_spade_rung0_v1.sh"
} >"$run_output/RUNG0_VERSIONS.txt"

run_arm() {
  local arm=$1
  local seed=$2
  local hint_path=${3:-}
  QWEN38_QUALIFICATION_OUTPUT_ROOT=$run_output \
  QWEN38_QUALIFICATION_AXES=$axes_csv \
  QWEN38_QUALIFICATION_NUM_TASKS=$tasks_per_axis \
  QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
  QWEN38_QUALIFICATION_MAX_CONCURRENT=$max_concurrent \
  QWEN38_QUALIFICATION_INDEX_OFFSET=200000 \
  QUALIFICATION_REASONING_EFFORT=$reasoning_effort \
  QUALIFICATION_SAMPLING_SEED=$seed \
  QUALIFICATION_SAMPLING_TEMPERATURE=$temperature \
  QUALIFICATION_PRIVILEGED_HINT_PATH=$hint_path \
  EVAL_PYTHON_BIN=$runtime_python \
  EVAL_CLIENT_BASE_URL=$base_url \
  scripts/run_qwen38_27b_prime_harness_qualification_v1.sh \
    "$model" "$arm-seed-$seed"
}

if [[ -n "$no_hint_source_root" ]]; then
  for seed in "${no_hint_seeds[@]}"; do
    source_dir=$no_hint_source_root/no-hint-seed-$seed
    target_dir=$run_output/no-hint-seed-$seed
    if [[ ! -f "$source_dir/SUMMARY.json" || ! -f "$source_dir/VERSIONS.txt" ]]; then
      echo "reusable no-hint source is incomplete: $source_dir" >&2
      exit 1
    fi
    source_versions=$source_dir/VERSIONS.txt
    for expected in \
      "model=$model" \
      "model_revision=$model_revision" \
      "axes=$axes_csv" \
      "num_tasks=$tasks_per_axis" \
      "num_rollouts=1" \
      "reasoning_effort=$reasoning_effort" \
      "index_offset=200000" \
      "sampling_seed=$seed" \
      "sampling_temperature=$temperature" \
      "privileged_hint_path=none"; do
      if ! grep -Fqx "$expected" "$source_versions"; then
        echo "reusable no-hint source mismatch: $expected" >&2
        exit 1
      fi
    done
    if [[ -e "$target_dir" ]]; then
      echo "refusing to overwrite no-hint target: $target_dir" >&2
      exit 1
    fi
    cp -a "$source_dir" "$target_dir"
    {
      printf 'reused_no_hint_seed=%s\n' "$seed"
      sha256sum "$source_versions" "$source_dir"/*/traces.jsonl
    } >>"$run_output/RUNG0_VERSIONS.txt"
  done
else
  for seed in "${no_hint_seeds[@]}"; do
    run_arm no-hint "$seed"
  done
fi
for seed in "${hinted_seeds[@]}"; do
  run_arm hinted "$seed" "$hint_artifact"
done

"$runtime_python" scripts/summarize_q35_2b_spade_rung0_v1.py \
  "$run_output" \
  --hints "$hint_artifact" \
  --expected-rollouts-per-arm "${#no_hint_seeds[@]}" \
  --output "$summary"
