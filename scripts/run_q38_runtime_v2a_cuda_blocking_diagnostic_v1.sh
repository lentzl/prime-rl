#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root=${RUNTIME_V2_ROOT:-/home/ubuntu/rlm/runtime-v2}
runtime_env=${RUNTIME_V2_ENV:-$runtime_root/envs/prime-rl-vllm-0.27.1-cu129}
output_root=${RUNTIME_V2_DIAGNOSTIC_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q38-runtime-v2-diagnostic-v1}
label=q38-runtime-v2a-cuda-blocking-3806011-r1
run_dir=$output_root/$label
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
  echo "refusing diagnostic while a GPU process is active" >&2
  exit 1
fi
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite diagnostic attempt: $run_dir" >&2
  exit 1
fi

started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
env \
  CUDA_LAUNCH_BLOCKING=1 \
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
run_status=$?
set -e
finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [[ ! -d "$run_dir" ]]; then
  echo "diagnostic produced no run directory: $run_dir" >&2
  exit 1
fi
{
  printf 'schema=q38-runtime-v2a-cuda-blocking-diagnostic/v1\n'
  printf 'admission_status=permanently_non_admitted\n'
  printf 'qualification_status=not_a_qualification_attempt\n'
  printf 'cuda_launch_blocking=1\n'
  printf 'baseline_exit_status=%s\n' "$run_status"
  printf 'started_at_utc=%s\n' "$started_at_utc"
  printf 'finished_at_utc=%s\n' "$finished_at_utc"
  printf 'vllm_version=%s\n' "$installed_version"
  printf 'teacher_model=%s\n' "$teacher_model"
  printf 'teacher_revision=%s\n' "$teacher_revision"
  printf 'task_index=3806011\n'
  printf 'tensor_parallel_size=2\n'
  printf 'dtype=bfloat16\n'
  printf 'max_model_len=32768\n'
  printf 'gpu_memory_utilization=0.88\n'
  printf 'enforce_eager=true\n'
  printf 'max_num_seqs=1\n'
  printf 'max_num_batched_tokens=512\n'
  printf 'disable_custom_all_reduce=true\n'
} >"$run_dir/DIAGNOSTIC-MANIFEST.txt"
find "$run_dir" -type f ! -name DIAGNOSTIC-SHA256SUMS.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"$run_dir/DIAGNOSTIC-SHA256SUMS.txt"

echo "CUDA-blocking diagnostic artifacts captured: $run_dir"
exit "$run_status"
