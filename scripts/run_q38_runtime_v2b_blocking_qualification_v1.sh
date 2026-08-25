#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root=${RUNTIME_V2_ROOT:-/home/ubuntu/rlm/runtime-v2}
runtime_env=${RUNTIME_V2_ENV:-$runtime_root/envs/prime-rl-vllm-0.27.1-cu129}
output_root=${RUNTIME_V2B_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q38-runtime-v2b-blocking-qualification-v1}
teacher_model=Qwen/Qwen3.8-27B
teacher_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
task_index=3806011
wrapper_path=$root/scripts/run_q38_runtime_v2b_blocking_qualification_v1.sh
wrapper_sha256=$(sha256sum "$wrapper_path" | awk '{print $1}')

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
if [[ ! -f "$runtime_root/RUNTIME-V2.txt" \
  || ! -f "$runtime_root/RUNTIME-V2.txt.sha256" ]]; then
  echo "runtime-v2 provenance manifest is missing" >&2
  exit 1
fi
(
  cd "$runtime_root"
  sha256sum -c RUNTIME-V2.txt.sha256
)
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing runtime-v2b qualification while a GPU process is active" >&2
  exit 1
fi
if [[ -e "$output_root" ]]; then
  echo "refusing to append to or overwrite runtime-v2b output: $output_root" >&2
  exit 1
fi
mkdir -p "$output_root"

attempt_failed=false
for attempt in 1 2 3; do
  label=q38-runtime-v2b-blocking-coldstart-3806011-r$attempt
  run_dir=$output_root/$label
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU process remained before runtime-v2b attempt $attempt" >&2
    attempt_failed=true
    break
  fi
  if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite runtime-v2b attempt: $run_dir" >&2
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
    QWEN38_QUALIFICATION_START_INDEX="$task_index" \
    QUALIFICATION_REASONING_EFFORT=xhigh \
    PRIME_MASTERY_OUTPUT_ROOT="$output_root" \
    "$root/scripts/run_qwen38_27b_prime_harness_baseline_v1.sh" \
    "$teacher_model" "$label" "$teacher_revision"
  run_status=$?
  set -e
  finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  if [[ ! -d "$run_dir" ]]; then
    echo "runtime-v2b attempt $attempt produced no run directory" >&2
    attempt_failed=true
    break
  fi

  set +e
  "$runtime_env/bin/python" - "$run_dir" \
    >"$run_dir/TRACE-VALIDATION.json" \
    2>"$run_dir/TRACE-VALIDATION.stderr" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
trace_files = list(run_dir.glob("natural_n1a_local/**/*.jsonl"))
envelopes = []
for trace_file in trace_files:
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            envelopes.append(json.loads(line))
if len(envelopes) != 1:
    raise SystemExit(f"expected one cold-start envelope, found {len(envelopes)}")
envelope = envelopes[0]
if envelope.get("ok") is not True or envelope.get("errors"):
    raise SystemExit("cold-start envelope contains a runtime error")
traces = envelope.get("traces") or []
if len(traces) != 1:
    raise SystemExit(f"expected one cold-start trace, found {len(traces)}")
trace = traces[0]
task_key = str((trace.get("task") or {}).get("key", ""))
if "03806011" not in task_key:
    raise SystemExit(f"unexpected cold-start task: {task_key}")
if trace.get("ok") is not True or trace.get("errors"):
    raise SystemExit("cold-start trace contains a runtime error")
if trace.get("is_completed") is not True:
    raise SystemExit("cold-start trace is incomplete")
print(
    json.dumps(
        {
            "envelope_id": envelope.get("id"),
            "trace_id": trace.get("id"),
            "task_key": task_key,
            "stop_condition": trace.get("stop_condition"),
            "harness_score": (trace.get("metrics") or {}).get("harness_score"),
            "final_answer_exact": (trace.get("metrics") or {}).get("final_answer_exact"),
            "model_calls": len(trace.get("calls") or []),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
  trace_status=$?
  set -e

  grep -En \
    'Triton kernel JIT compilation during inference:' \
    "$run_dir/inference.log" >"$run_dir/JIT-WARNINGS.txt" || true
  set +e
  grep -REni --include='*.log' \
    'ProviderError|illegal memory access|illegal instruction|torch\.AcceleratorError|CUDA error|RPC call .*timed out|RPC.*timeout|EngineCore.*(failed|died)|Engine core proc failed|inference exited or became unhealthy' \
    "$run_dir" >"$run_dir/FORBIDDEN-INFRASTRUCTURE-SIGNATURES.txt"
  grep_status=$?
  set -e
  if [[ $grep_status -eq 0 ]]; then
    forbidden_status=1
  elif [[ $grep_status -eq 1 ]]; then
    forbidden_status=0
  else
    forbidden_status=1
  fi

  teardown_status=0
  for _ in $(seq 1 30); do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] \
      && ! pgrep -f "$run_dir/inference.toml" >/dev/null; then
      break
    fi
    sleep 1
  done
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] \
    || pgrep -f "$run_dir/inference.toml" >/dev/null; then
    teardown_status=1
  fi

  {
    printf 'schema=q38-runtime-v2b-blocking-coldstart/v1\n'
    printf 'runtime_lane=runtime-v2b-blocking\n'
    printf 'teacher_admission_status=permanently_non_admitted\n'
    printf 'runtime_qualification_attempt=true\n'
    printf 'attempt=%s\n' "$attempt"
    printf 'cuda_launch_blocking=1\n'
    printf 'baseline_exit_status=%s\n' "$run_status"
    printf 'trace_validation_status=%s\n' "$trace_status"
    printf 'forbidden_signature_status=%s\n' "$forbidden_status"
    printf 'teardown_status=%s\n' "$teardown_status"
    printf 'started_at_utc=%s\n' "$started_at_utc"
    printf 'finished_at_utc=%s\n' "$finished_at_utc"
    printf 'vllm_version=%s\n' "$installed_version"
    printf 'teacher_model=%s\n' "$teacher_model"
    printf 'teacher_revision=%s\n' "$teacher_revision"
    printf 'task_index=%s\n' "$task_index"
    printf 'tensor_parallel_size=2\n'
    printf 'dtype=bfloat16\n'
    printf 'max_model_len=32768\n'
    printf 'gpu_memory_utilization=0.88\n'
    printf 'enforce_eager=true\n'
    printf 'max_num_seqs=1\n'
    printf 'max_num_batched_tokens=512\n'
    printf 'disable_custom_all_reduce=true\n'
    printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
  } >"$run_dir/RUNTIME-V2B-ATTEMPT.txt"
  find "$run_dir" -type f ! -name RUNTIME-V2B-SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"$run_dir/RUNTIME-V2B-SHA256SUMS.txt"

  if [[ $run_status -ne 0 || $trace_status -ne 0 \
    || $forbidden_status -ne 0 || $teardown_status -ne 0 ]]; then
    echo "runtime-v2b attempt $attempt failed infrastructure qualification" >&2
    attempt_failed=true
    break
  fi
done

if [[ "$attempt_failed" == true ]]; then
  echo "runtime-v2b rejected; stopping without a follow-on action" >&2
  exit 1
fi

qualification_manifest=$output_root/RUNTIME-V2B-QUALIFIED.txt
{
  printf 'schema=q38-runtime-v2b-blocking-qualification/v1\n'
  printf 'status=qualified\n'
  printf 'runtime_lane=runtime-v2b-blocking\n'
  printf 'cuda_launch_blocking=1\n'
  printf 'teacher_admission_status=qualification_attempts_permanently_non_admitted\n'
  printf 'vllm_version=%s\n' "$installed_version"
  printf 'teacher_model=%s\n' "$teacher_model"
  printf 'teacher_revision=%s\n' "$teacher_revision"
  printf 'task_index=%s\n' "$task_index"
  printf 'independent_fresh_server_starts=3\n'
  printf 'qualified_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
  sha256sum "$output_root"/q38-runtime-v2b-blocking-coldstart-3806011-r*/RUNTIME-V2B-SHA256SUMS.txt
} >"$qualification_manifest"
sha256sum "$qualification_manifest" >"$qualification_manifest.sha256"
echo "runtime-v2b blocking cold-start qualification passed"
