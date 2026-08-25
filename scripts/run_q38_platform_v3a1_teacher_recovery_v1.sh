#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_env=${PLATFORM_V3_RUNTIME_ENV:-$root/.venv}
platform_root=${PLATFORM_V3_ROOT:-/home/ubuntu/rlm/platform-v3-a6000x2}
qualification_root=${PLATFORM_V3_QUALIFICATION_ROOT:-/home/ubuntu/rlm/results/q38-platform-v3a1-a6000x2-vllm026-qualification-v1}
output_root=${PLATFORM_V3_RECOVERY_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q38-platform-v3a1-teacher-recovery-v1}
teacher_model=Qwen/Qwen3.8-27B
teacher_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
qualification_manifest=$qualification_root/PLATFORM-V3A1-QUALIFIED.txt
proposal=$root/experiments/qwen38-to-qwen35-2b-role-distillation-v1/platform-v3-a6000x2-proposal.json
host_manifest=$platform_root/HOST-MANIFEST.txt
runtime_manifest=$platform_root/RUNTIME-MANIFEST.txt
wrapper_path=$root/scripts/run_q38_platform_v3a1_teacher_recovery_v1.sh
wrapper_sha256=$(sha256sum "$wrapper_path" | awk '{print $1}')
uv_bin=${UV_BIN:-$(command -v uv || true)}

if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin=$HOME/.local/bin/uv
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv executable not found" >&2
  exit 1
fi
for executable in python inference eval vllm-router; do
  if [[ ! -x "$runtime_env/bin/$executable" ]]; then
    echo "platform-v3 runtime executable is missing: $runtime_env/bin/$executable" >&2
    exit 1
  fi
done
for required_file in \
  "$proposal" \
  "$host_manifest" "$host_manifest.sha256" \
  "$runtime_manifest" "$runtime_manifest.sha256" \
  "$qualification_manifest" "$qualification_manifest.sha256"; do
  if [[ ! -f "$required_file" ]]; then
    echo "platform-v3 recovery provenance file is missing: $required_file" >&2
    exit 1
  fi
done
(
  cd "$platform_root"
  sha256sum -c HOST-MANIFEST.txt.sha256
  sha256sum -c RUNTIME-MANIFEST.txt.sha256
)
(
  cd "$qualification_root"
  sha256sum -c PLATFORM-V3A1-QUALIFIED.txt.sha256
)

installed_version=$(
  "$uv_bin" run --no-sync --python "$runtime_env/bin/python" python - <<'PY'
import importlib.metadata

print(importlib.metadata.version("vllm"))
PY
)
if [[ "$installed_version" != "0.26.0+cu129" ]]; then
  echo "platform-v3a1 has unexpected vLLM version: $installed_version" >&2
  exit 1
fi
for expected_line in \
  'status=qualified' \
  'platform_lane=platform-v3a1-a6000x2-vllm026' \
  'teacher_admission_status=qualification_permanently_non_admitted' \
  'fresh_server_lifetimes=3' \
  'sentinel_traces=1' \
  'composite_workload_traces_per_restart=9' \
  'vllm_version=0.26.0+cu129' \
  'teacher_model=Qwen/Qwen3.8-27B' \
  'teacher_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'; do
  if ! grep -Fqx "$expected_line" "$qualification_manifest"; then
    echo "qualification manifest is missing expected line: $expected_line" >&2
    exit 1
  fi
done
qualification_manifest_sha256=$(sha256sum "$qualification_manifest" | awk '{print $1}')

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
  local forbidden_scan_status
  local forbidden_status
  local teardown_status
  local forbidden_tmp

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
  "$uv_bin" run --no-sync --python "$runtime_env/bin/python" python - \
    "$run_dir" "$axis" "$start_index" "$draws" \
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
trace_files = sorted(run_dir.glob(f"{axis}/**/traces.jsonl"))
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
for trace in traces:
    metrics = trace.get("metrics") or {}
    records.append(
        {
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

  sudo journalctl -k --since "$started_at_utc" --until "$finished_at_utc" --no-pager \
    >"$run_dir/KERNEL-JOURNAL.txt" 2>&1 || true
  forbidden_tmp=$(mktemp "${TMPDIR:-/tmp}/q38-platform-v3-recovery-forbidden.XXXXXX")
  set +e
  grep -REni --include='*.log' --include='*.txt' \
    'ProviderError|illegal memory access|illegal instruction|torch\.AcceleratorError|CUDA error|CUBLAS_STATUS|RPC call .*timed out|RPC.*timeout|EngineCore.*(failed|died)|Engine core proc failed|inference exited or became unhealthy|NVRM: Xid|Xid \([0-9]+\)' \
    "$run_dir" >"$forbidden_tmp"
  forbidden_scan_status=$?
  set -e
  mv "$forbidden_tmp" "$run_dir/FORBIDDEN-INFRASTRUCTURE-SIGNATURES.txt"
  if [[ $forbidden_scan_status -eq 0 ]]; then
    forbidden_status=1
  elif [[ $forbidden_scan_status -eq 1 ]]; then
    forbidden_status=0
  else
    forbidden_status=1
  fi

  teardown_status=0
  for _ in $(seq 1 60); do
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
    printf 'schema=q38-platform-v3a1-teacher-recovery-leg/v1\n'
    printf 'platform_lane=platform-v3a1-a6000x2-vllm026\n'
    printf 'teacher_admission_status=subject_to_hard_and_provenance_gates\n'
    printf 'axis=%s\n' "$axis"
    printf 'start_index=%s\n' "$start_index"
    printf 'draws=%s\n' "$draws"
    printf 'cuda_launch_blocking=unset\n'
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
  } >"$run_dir/PLATFORM-V3A1-RECOVERY-LEG.txt"
  find "$run_dir" -type f ! -name PLATFORM-V3A1-RECOVERY-SHA256SUMS.txt -print0 \
    | sort -z | xargs -0 sha256sum \
    >"$run_dir/PLATFORM-V3A1-RECOVERY-SHA256SUMS.txt"

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
  q38-platform-v3a1-n1a-local-3806011-5-r1

run_leg \
  natural_n1b \
  3805000 \
  16 \
  q38-platform-v3a1-n1b-principal16-r1

recovery_manifest=$output_root/PLATFORM-V3A1-TEACHER-RECOVERY-COMPLETE.txt
{
  printf 'schema=q38-platform-v3a1-teacher-recovery/v1\n'
  printf 'status=infrastructure_complete_not_admitted\n'
  printf 'platform_lane=platform-v3a1-a6000x2-vllm026\n'
  printf 'cuda_launch_blocking=unset\n'
  printf 'teacher_model=%s\n' "$teacher_model"
  printf 'teacher_revision=%s\n' "$teacher_revision"
  printf 'runtime_qualification_manifest_sha256=%s\n' "$qualification_manifest_sha256"
  printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
  sha256sum "$output_root"/*/PLATFORM-V3A1-RECOVERY-SHA256SUMS.txt
} >"$recovery_manifest"
sha256sum "$recovery_manifest" >"$recovery_manifest.sha256"
echo "platform-v3a1 teacher recovery completed; traces remain unadmitted"
