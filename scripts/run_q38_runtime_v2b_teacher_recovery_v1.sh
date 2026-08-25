#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root=${RUNTIME_V2_ROOT:-/home/ubuntu/rlm/runtime-v2}
runtime_env=${RUNTIME_V2_ENV:-$runtime_root/envs/prime-rl-vllm-0.27.1-cu129}
qualification_root=${RUNTIME_V2B_QUALIFICATION_ROOT:-/home/ubuntu/rlm/results/q38-runtime-v2b-blocking-qualification-v1}
output_root=${RUNTIME_V2B_RECOVERY_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q38-runtime-v2b-teacher-recovery-v1}
teacher_model=Qwen/Qwen3.8-27B
teacher_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
qualification_manifest=$qualification_root/RUNTIME-V2B-QUALIFIED.txt
qualification_manifest_sha256=d827b6a1a9fc25b92bd79e46d89e3fb82737c84a34ed77a76ce22e134e9e1992
wrapper_path=$root/scripts/run_q38_runtime_v2b_teacher_recovery_v1.sh
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
if [[ ! -f "$qualification_manifest" \
  || ! -f "$qualification_manifest.sha256" ]]; then
  echo "runtime-v2b qualification manifest is missing" >&2
  exit 1
fi
actual_qualification_sha256=$(sha256sum "$qualification_manifest" | awk '{print $1}')
if [[ "$actual_qualification_sha256" != "$qualification_manifest_sha256" ]]; then
  echo "runtime-v2b qualification manifest hash changed" >&2
  exit 1
fi
sha256sum -c "$qualification_manifest.sha256"
if ! grep -qx 'status=qualified' "$qualification_manifest" \
  || ! grep -qx 'runtime_lane=runtime-v2b-blocking' "$qualification_manifest" \
  || ! grep -qx 'cuda_launch_blocking=1' "$qualification_manifest"; then
  echo "runtime-v2b qualification manifest has unexpected contents" >&2
  exit 1
fi
(
  cd "$runtime_root"
  sha256sum -c RUNTIME-V2.txt.sha256
)
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing teacher recovery while a GPU process is active" >&2
  exit 1
fi
if [[ -e "$output_root" ]]; then
  echo "refusing to append to or overwrite teacher recovery: $output_root" >&2
  exit 1
fi
mkdir -p "$output_root"

run_leg() {
  local axis=$1
  local start_index=$2
  local draws=$3
  local label=$4
  local run_dir=$output_root/$label
  local started_at_utc
  local finished_at_utc
  local run_status
  local audit_status
  local grep_status
  local forbidden_status
  local teardown_status

  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU process remained before teacher-recovery leg $label" >&2
    return 1
  fi
  if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite teacher-recovery leg: $run_dir" >&2
    return 1
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
    echo "teacher-recovery leg produced no run directory: $label" >&2
    return 1
  fi

  set +e
  "$runtime_env/bin/python" - "$run_dir" "$axis" "$start_index" "$draws" \
    >"$run_dir/RECOVERY-AUDIT.json" \
    2>"$run_dir/RECOVERY-AUDIT.stderr" <<'PY'
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
traces = [trace for envelope in envelopes for trace in (envelope.get("traces") or [])]
if len(envelopes) != draws or len(traces) != draws:
    raise SystemExit(
        f"expected {draws} envelopes/traces, found {len(envelopes)}/{len(traces)}"
    )
for envelope in envelopes:
    if envelope.get("ok") is not True or envelope.get("errors"):
        raise SystemExit("recovery envelope contains a runtime error")
    if len(envelope.get("traces") or []) != 1:
        raise SystemExit("recovery envelope does not contain exactly one trace")
for trace in traces:
    if trace.get("ok") is not True or trace.get("errors"):
        raise SystemExit("recovery trace contains a runtime error")
    if trace.get("is_completed") is not True:
        raise SystemExit("recovery trace is incomplete")
task_keys = [str((trace.get("task") or {}).get("key", "")) for trace in traces]
indices = []
for task_key in task_keys:
    match = re.search(r"-(\d{8})-", task_key)
    if match is None or axis not in task_key:
        raise SystemExit(f"unexpected task key: {task_key}")
    indices.append(int(match.group(1)))
expected_indices = list(range(start_index, start_index + draws))
if sorted(indices) != expected_indices:
    raise SystemExit(f"unexpected task indices: {sorted(indices)} != {expected_indices}")
records = []
for envelope, trace in zip(envelopes, traces, strict=True):
    metrics = trace.get("metrics") or {}
    records.append(
        {
            "envelope_id": envelope.get("id"),
            "trace_id": trace.get("id"),
            "task_key": (trace.get("task") or {}).get("key"),
            "stop_condition": trace.get("stop_condition"),
            "harness_score": metrics.get("harness_score"),
            "final_answer_exact": metrics.get("final_answer_exact"),
            "model_calls": len(trace.get("calls") or []),
        }
    )
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
    printf 'schema=q38-runtime-v2b-teacher-recovery-leg/v1\n'
    printf 'runtime_lane=runtime-v2b-blocking\n'
    printf 'teacher_admission_status=subject_to_hard_and_provenance_gates\n'
    printf 'axis=%s\n' "$axis"
    printf 'start_index=%s\n' "$start_index"
    printf 'draws=%s\n' "$draws"
    printf 'cuda_launch_blocking=1\n'
    printf 'baseline_exit_status=%s\n' "$run_status"
    printf 'recovery_audit_status=%s\n' "$audit_status"
    printf 'forbidden_signature_status=%s\n' "$forbidden_status"
    printf 'teardown_status=%s\n' "$teardown_status"
    printf 'started_at_utc=%s\n' "$started_at_utc"
    printf 'finished_at_utc=%s\n' "$finished_at_utc"
    printf 'vllm_version=%s\n' "$installed_version"
    printf 'teacher_model=%s\n' "$teacher_model"
    printf 'teacher_revision=%s\n' "$teacher_revision"
    printf 'runtime_qualification_manifest_sha256=%s\n' "$qualification_manifest_sha256"
    printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
  } >"$run_dir/RUNTIME-V2B-RECOVERY-LEG.txt"
  find "$run_dir" -type f ! -name RUNTIME-V2B-RECOVERY-SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"$run_dir/RUNTIME-V2B-RECOVERY-SHA256SUMS.txt"

  if [[ $run_status -ne 0 || $audit_status -ne 0 \
    || $forbidden_status -ne 0 || $teardown_status -ne 0 ]]; then
    echo "teacher-recovery leg $label failed infrastructure validation" >&2
    return 1
  fi
}

run_leg \
  natural_n1a_local \
  3806011 \
  5 \
  q38-runtime-v2b-n1a-local-3806011-5-r1

run_leg \
  natural_n1b \
  3805000 \
  16 \
  q38-runtime-v2b-n1b-principal16-r1

recovery_manifest=$output_root/RUNTIME-V2B-TEACHER-RECOVERY-COMPLETE.txt
{
  printf 'schema=q38-runtime-v2b-teacher-recovery/v1\n'
  printf 'status=infrastructure_complete\n'
  printf 'runtime_lane=runtime-v2b-blocking\n'
  printf 'cuda_launch_blocking=1\n'
  printf 'teacher_model=%s\n' "$teacher_model"
  printf 'teacher_revision=%s\n' "$teacher_revision"
  printf 'runtime_qualification_manifest_sha256=%s\n' "$qualification_manifest_sha256"
  printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
  sha256sum "$output_root"/*/RUNTIME-V2B-RECOVERY-SHA256SUMS.txt
} >"$recovery_manifest"
sha256sum "$recovery_manifest" >"$recovery_manifest.sha256"
echo "runtime-v2b teacher recovery completed"
