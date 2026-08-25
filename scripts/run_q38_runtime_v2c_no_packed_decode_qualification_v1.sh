#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root=${RUNTIME_V2_ROOT:-/home/ubuntu/rlm/runtime-v2}
runtime_env=${RUNTIME_V2_ENV:-$runtime_root/envs/prime-rl-vllm-0.27.1-cu129}
output_root=${RUNTIME_V2C_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q38-runtime-v2c-no-packed-decode-qualification-v1}
teacher_model=Qwen/Qwen3.8-27B
teacher_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
axis=natural_n1a_local
sentinel_index=3806011
soak_draws=5
wrapper_path=$root/scripts/run_q38_runtime_v2c_no_packed_decode_qualification_v1.sh
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

flag_value=$(
  env VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 \
    "$runtime_env/bin/python" -c \
    'from vllm import envs; print(int(envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE))'
)
if [[ "$flag_value" != "0" ]]; then
  echo "vLLM did not resolve the packed recurrent decode flag to false" >&2
  exit 1
fi

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing runtime-v2c qualification while a GPU process is active" >&2
  exit 1
fi
if [[ -e "$output_root" ]]; then
  echo "refusing to append to or overwrite runtime-v2c output: $output_root" >&2
  exit 1
fi
mkdir -p "$output_root"

write_terminal_manifest() {
  local status=$1
  local failed_leg=${2:-none}
  local terminal_manifest=$output_root/RUNTIME-V2C-${status^^}.txt
  {
    printf 'schema=q38-runtime-v2c-no-packed-decode-terminal/v1\n'
    printf 'status=%s\n' "$status"
    printf 'failed_leg=%s\n' "$failed_leg"
    printf 'runtime_lane=runtime-v2c-no-packed-decode\n'
    printf 'cuda_launch_blocking=1\n'
    printf 'vllm_enable_fla_packed_recurrent_decode=0\n'
    printf 'teacher_admission_status=all_qualification_traces_permanently_non_admitted\n'
    printf 'vllm_version=%s\n' "$installed_version"
    printf 'teacher_model=%s\n' "$teacher_model"
    printf 'teacher_revision=%s\n' "$teacher_revision"
    printf 'sentinel_task_index=%s\n' "$sentinel_index"
    printf 'soak_draws=%s\n' "$soak_draws"
    printf 'finished_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
    find "$output_root" -mindepth 2 -maxdepth 2 \
      -name RUNTIME-V2C-SHA256SUMS.txt -print0 \
      | sort -z \
      | xargs -0 -r sha256sum
  } >"$terminal_manifest"
  sha256sum "$terminal_manifest" >"$terminal_manifest.sha256"
}

run_leg() {
  local phase=$1
  local attempt=$2
  local start_index=$3
  local draws=$4
  local label=q38-runtime-v2c-${phase}-3806011-r${attempt}
  local run_dir=$output_root/$label
  local started_at_utc
  local finished_at_utc
  local run_status
  local audit_status
  local grep_status
  local forbidden_status
  local intervention_status
  local teardown_status

  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU process remained before runtime-v2c leg $label" >&2
    return 1
  fi
  if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite runtime-v2c leg: $run_dir" >&2
    return 1
  fi

  started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  set +e
  env \
    CUDA_LAUNCH_BLOCKING=1 \
    VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 \
    INFERENCE_BIN="$runtime_env/bin/inference" \
    EVAL_BIN="$runtime_env/bin/eval" \
    EVAL_PYTHON_BIN="$runtime_env/bin/python" \
    VLLM_ROUTER_BIN="$runtime_env/bin/vllm-router" \
    EVAL_MAX_NUM_BATCHED_TOKENS=512 \
    EVAL_MAX_NUM_SEQS=1 \
    EVAL_DISABLE_CUSTOM_ALL_REDUCE=true \
    QWEN38_QUALIFICATION_AXES="$axis" \
    QWEN38_QUALIFICATION_NUM_TASKS="$draws" \
    QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
    QWEN38_QUALIFICATION_MAX_CONCURRENT=1 \
    QWEN38_QUALIFICATION_START_INDEX="$start_index" \
    QUALIFICATION_REASONING_EFFORT=xhigh \
    PRIME_MASTERY_OUTPUT_ROOT="$output_root" \
    "$root/scripts/run_qwen38_27b_prime_harness_baseline_v1.sh" \
    "$teacher_model" "$label" "$teacher_revision"
  run_status=$?
  set -e
  finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  if [[ ! -d "$run_dir" ]]; then
    echo "runtime-v2c leg produced no run directory: $label" >&2
    return 1
  fi

  set +e
  "$runtime_env/bin/python" - "$run_dir" "$axis" "$start_index" "$draws" \
    >"$run_dir/TRACE-VALIDATION.json" \
    2>"$run_dir/TRACE-VALIDATION.stderr" <<'PY'
import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
axis = sys.argv[2]
start_index = int(sys.argv[3])
draws = int(sys.argv[4])
trace_files = list(run_dir.glob(f"{axis}/**/traces.jsonl"))
envelopes = []
for trace_file in trace_files:
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            envelopes.append(json.loads(line))

traces = []
for envelope in envelopes:
    if envelope.get("ok") is not True or envelope.get("errors"):
        raise SystemExit("qualification envelope contains a runtime error")
    envelope_traces = envelope.get("traces") or []
    if len(envelope_traces) != 1:
        raise SystemExit("qualification envelope does not contain exactly one trace")
    traces.extend(envelope_traces)

if len(envelopes) != draws or len(traces) != draws:
    raise SystemExit(
        f"expected {draws} envelopes/traces, found {len(envelopes)}/{len(traces)}"
    )

records = []
indices = []
for envelope, trace in zip(envelopes, traces, strict=True):
    if trace.get("ok") is not True or trace.get("errors"):
        raise SystemExit("qualification trace contains a runtime error")
    if trace.get("is_completed") is not True:
        raise SystemExit("qualification trace is incomplete")
    task_key = str((trace.get("task") or {}).get("key", ""))
    match = re.search(r"-(\d{8})-", task_key)
    if match is None or axis not in task_key:
        raise SystemExit(f"unexpected task key: {task_key}")
    indices.append(int(match.group(1)))
    metrics = trace.get("metrics") or {}
    records.append(
        {
            "envelope_id": envelope.get("id"),
            "trace_id": trace.get("id"),
            "task_key": task_key,
            "stop_condition": trace.get("stop_condition"),
            "harness_score": metrics.get("harness_score"),
            "final_answer_exact": metrics.get("final_answer_exact"),
            "model_calls": len(trace.get("calls") or []),
        }
    )

expected_indices = list(range(start_index, start_index + draws))
if sorted(indices) != expected_indices:
    raise SystemExit(f"unexpected task indices: {sorted(indices)} != {expected_indices}")

print(
    json.dumps(
        {
            "axis": axis,
            "draws": draws,
            "infrastructure_complete": len(records),
            "hard_successes": sum(r["harness_score"] == 1.0 for r in records),
            "records": records,
            "start_index": start_index,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
  audit_status=$?
  set -e

  grep -En \
    'Triton kernel JIT compilation during inference:' \
    "$run_dir/inference.log" >"$run_dir/JIT-WARNINGS.txt" || true
  set +e
  grep -REni --include='*.log' \
    'ProviderError|illegal memory access|illegal instruction|torch\.AcceleratorError|CUDA error|CUBLAS_STATUS|RPC call .*timed out|RPC.*timeout|EngineCore.*(failed|died)|Engine core proc failed|inference exited or became unhealthy' \
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

  intervention_status=0
  if grep -q 'fused_recurrent_gated_delta_rule_packed_decode_kernel' \
    "$run_dir/inference.log"; then
    echo "packed recurrent decode kernel appeared despite runtime-v2c flag" \
      >"$run_dir/INTERVENTION-VALIDATION.stderr"
    intervention_status=1
  else
    printf '%s\n' \
      'VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0; packed decode JIT signature absent' \
      >"$run_dir/INTERVENTION-VALIDATION.txt"
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
    printf 'schema=q38-runtime-v2c-no-packed-decode-leg/v1\n'
    printf 'runtime_lane=runtime-v2c-no-packed-decode\n'
    printf 'teacher_admission_status=permanently_non_admitted\n'
    printf 'runtime_qualification_attempt=true\n'
    printf 'phase=%s\n' "$phase"
    printf 'attempt=%s\n' "$attempt"
    printf 'axis=%s\n' "$axis"
    printf 'start_index=%s\n' "$start_index"
    printf 'draws=%s\n' "$draws"
    printf 'cuda_launch_blocking=1\n'
    printf 'vllm_enable_fla_packed_recurrent_decode=0\n'
    printf 'baseline_exit_status=%s\n' "$run_status"
    printf 'trace_validation_status=%s\n' "$audit_status"
    printf 'forbidden_signature_status=%s\n' "$forbidden_status"
    printf 'intervention_validation_status=%s\n' "$intervention_status"
    printf 'teardown_status=%s\n' "$teardown_status"
    printf 'started_at_utc=%s\n' "$started_at_utc"
    printf 'finished_at_utc=%s\n' "$finished_at_utc"
    printf 'vllm_version=%s\n' "$installed_version"
    printf 'teacher_model=%s\n' "$teacher_model"
    printf 'teacher_revision=%s\n' "$teacher_revision"
    printf 'tensor_parallel_size=2\n'
    printf 'dtype=bfloat16\n'
    printf 'max_model_len=32768\n'
    printf 'gpu_memory_utilization=0.88\n'
    printf 'enforce_eager=true\n'
    printf 'max_num_seqs=1\n'
    printf 'max_num_batched_tokens=512\n'
    printf 'disable_custom_all_reduce=true\n'
    printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
  } >"$run_dir/RUNTIME-V2C-LEG.txt"
  find "$run_dir" -type f ! -name RUNTIME-V2C-SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"$run_dir/RUNTIME-V2C-SHA256SUMS.txt"

  if [[ $run_status -ne 0 || $audit_status -ne 0 \
    || $forbidden_status -ne 0 || $intervention_status -ne 0 \
    || $teardown_status -ne 0 ]]; then
    echo "runtime-v2c leg $label failed infrastructure qualification" >&2
    return 1
  fi
}

for attempt in 1 2 3; do
  if ! run_leg sentinel "$attempt" "$sentinel_index" 1; then
    write_terminal_manifest rejected "sentinel-r$attempt"
    exit 1
  fi
done

if ! run_leg soak 1 "$sentinel_index" "$soak_draws"; then
  write_terminal_manifest rejected soak-r1
  exit 1
fi

# The baseline wrapper owns one server lifetime. Completing soak-r1 teardown
# therefore makes soak-r2 an explicit fresh restart.
if ! run_leg soak 2 "$sentinel_index" "$soak_draws"; then
  write_terminal_manifest rejected soak-r2
  exit 1
fi

write_terminal_manifest qualified none
echo "runtime-v2c no-packed-decode qualification passed"
