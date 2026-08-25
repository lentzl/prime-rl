#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root=${RUNTIME_V2_ROOT:-/home/ubuntu/rlm/runtime-v2}
runtime_env=${RUNTIME_V2_ENV:-$runtime_root/envs/prime-rl-vllm-0.27.1-cu129}
output_root=${RUNTIME_V2_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q38-runtime-v2-qualification-v1}
teacher_model=Qwen/Qwen3.8-27B
teacher_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

for executable in python inference eval vllm-router; do
  if [[ ! -x "$runtime_env/bin/$executable" ]]; then
    echo "runtime-v2 executable is missing: $runtime_env/bin/$executable" >&2
    exit 1
  fi
done
installed_version=$(
  "$runtime_env/bin/python" -c \
    'import importlib.metadata; print(importlib.metadata.version("vllm"))'
)
if [[ "$installed_version" != "0.27.1+cu129" ]]; then
  echo "runtime-v2 has unexpected vLLM version: $installed_version" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing cold-start qualification while a GPU process is active" >&2
  exit 1
fi

mkdir -p "$output_root"
for attempt in 1 2 3; do
  label=q38-runtime-v2-coldstart-3806011-r$attempt
  run_dir=$output_root/$label
  if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite cold-start attempt: $run_dir" >&2
    exit 1
  fi

  env \
    INFERENCE_BIN="$runtime_env/bin/inference" \
    EVAL_BIN="$runtime_env/bin/eval" \
    EVAL_PYTHON_BIN="$runtime_env/bin/python" \
    VLLM_ROUTER_BIN="$runtime_env/bin/vllm-router" \
    EVAL_MAX_NUM_BATCHED_TOKENS=512 \
    EVAL_MAX_NUM_SEQS=1 \
    EVAL_DISABLE_CUSTOM_ALL_REDUCE=true \
    QWEN38_QUALIFICATION_AXES=natural_n1a_local \
    QWEN38_QUALIFICATION_NUM_TASKS=1 \
    QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
    QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
    QWEN38_QUALIFICATION_START_INDEX=3806011 \
    QUALIFICATION_REASONING_EFFORT=xhigh \
    PRIME_MASTERY_OUTPUT_ROOT="$output_root" \
    "$root/scripts/run_qwen38_27b_prime_harness_baseline_v1.sh" \
    "$teacher_model" "$label" "$teacher_revision"

  "$runtime_env/bin/python" - "$run_dir" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
trace_files = list(run_dir.glob("natural_n1a_local/**/*.jsonl"))
episodes = []
for trace_file in trace_files:
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            episodes.append(json.loads(line))
if len(episodes) != 1:
    raise SystemExit(f"expected one cold-start episode, found {len(episodes)}")
traces = episodes[0].get("traces") or []
if len(traces) != 1:
    raise SystemExit(f"expected one cold-start trace, found {len(traces)}")
trace = traces[0]
task_key = str((trace.get("task") or {}).get("key", ""))
if "03806011" not in task_key:
    raise SystemExit(f"unexpected cold-start task: {task_key}")
if trace.get("ok") is not True or trace.get("errors"):
    raise SystemExit("cold-start trace contains a ProviderError or other runtime error")
PY

  if grep -Eni \
    'illegal memory access|illegal instruction|RPC call .*timed out|EngineCore.*(failed|died|exited)|Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel' \
    "$run_dir/inference.log"; then
    echo "runtime-v2 cold-start attempt $attempt hit a forbidden infrastructure signature" >&2
    exit 1
  fi
  sha256sum \
    "$run_dir/inference.log" \
    "$run_dir/inference.toml" \
    "$run_dir/VERSIONS.txt" \
    >"$run_dir/COLDSTART-SHA256SUMS.txt"
done

qualification_manifest=$output_root/RUNTIME-V2-QUALIFIED.txt
if [[ -e "$qualification_manifest" ]]; then
  echo "refusing to overwrite qualification manifest: $qualification_manifest" >&2
  exit 1
fi
{
  printf 'schema=q38-runtime-v2-coldstart/v1\n'
  printf 'status=qualified\n'
  printf 'vllm_version=%s\n' "$installed_version"
  printf 'teacher_model=%s\n' "$teacher_model"
  printf 'teacher_revision=%s\n' "$teacher_revision"
  printf 'task_index=3806011\n'
  printf 'independent_cold_starts=3\n'
  printf 'qualified_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "$output_root"/q38-runtime-v2-coldstart-3806011-r*/COLDSTART-SHA256SUMS.txt
} >"$qualification_manifest"
sha256sum "$qualification_manifest" >"$qualification_manifest.sha256"
echo "runtime-v2 cold-start qualification passed"
