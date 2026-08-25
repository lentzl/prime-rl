#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_env=${PLATFORM_V3_RUNTIME_ENV:-$root/.venv}
platform_root=${PLATFORM_V3_ROOT:-/home/ubuntu/rlm/platform-v3-a6000x2}
output_root=${PLATFORM_V3_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q38-platform-v3a1-a6000x2-vllm026-qualification-v1}
teacher_model=Qwen/Qwen3.8-27B
teacher_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
proposal=$root/experiments/qwen38-to-qwen35-2b-role-distillation-v1/platform-v3-a6000x2-proposal.json
host_manifest=$platform_root/HOST-MANIFEST.txt
wrapper_path=$root/scripts/run_q38_platform_v3a_a6000_qualification_v1.sh
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
for required_file in "$proposal" "$host_manifest" "$host_manifest.sha256"; do
  if [[ ! -f "$required_file" ]]; then
    echo "platform-v3 provenance file is missing: $required_file" >&2
    exit 1
  fi
done
(
  cd "$platform_root"
  sha256sum -c HOST-MANIFEST.txt.sha256
)

installed_version=$(
  "$uv_bin" run --no-sync --python "$runtime_env/bin/python" python - <<'PY'
import importlib.metadata

print(importlib.metadata.version("vllm"))
PY
)
if [[ "$installed_version" != "0.26.0+cu129" ]]; then
  echo "platform-v3a has unexpected vLLM version: $installed_version" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing qualification while a GPU process is active" >&2
  exit 1
fi
if [[ -e "$output_root" ]]; then
  echo "refusing to append to or overwrite platform-v3 output: $output_root" >&2
  exit 1
fi
mkdir -p "$output_root"

run_leg() {
  local label=$1
  local expected_traces=$2
  local eval_driver=$3
  local axes=$4
  local num_tasks=$5
  local start_index=$6
  local run_dir=$output_root/$label
  local started_at_utc
  local finished_at_utc
  local run_status
  local validation_status
  local forbidden_status
  local teardown_status

  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU process remained before $label" >&2
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
    EVAL_DRIVER="$eval_driver" \
    QWEN38_QUALIFICATION_AXES="$axes" \
    QWEN38_QUALIFICATION_NUM_TASKS="$num_tasks" \
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
    echo "$label produced no run directory" >&2
    return 1
  fi

  set +e
  "$uv_bin" run --no-sync --python "$runtime_env/bin/python" python - \
    "$run_dir" "$expected_traces" >"$run_dir/TRACE-VALIDATION.json" \
    2>"$run_dir/TRACE-VALIDATION.stderr" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
expected_traces = int(sys.argv[2])
envelopes = []
for trace_file in run_dir.glob("**/*.jsonl"):
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            envelopes.append(json.loads(line))
traces = []
for envelope in envelopes:
    if envelope.get("ok") is not True or envelope.get("errors"):
        raise SystemExit("qualification envelope contains a runtime error")
    traces.extend(envelope.get("traces") or [])
if len(traces) != expected_traces:
    raise SystemExit(
        f"expected {expected_traces} qualification traces, found {len(traces)}"
    )
for trace in traces:
    if trace.get("ok") is not True or trace.get("errors"):
        raise SystemExit("qualification trace contains a runtime error")
    if trace.get("is_completed") is not True:
        raise SystemExit("qualification trace is incomplete")
print(
    json.dumps(
        {
            "envelopes": len(envelopes),
            "traces": len(traces),
            "task_keys": [str((trace.get("task") or {}).get("key", "")) for trace in traces],
            "model_calls": sum(len(trace.get("calls") or []) for trace in traces),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
  validation_status=$?
  set -e

  sudo journalctl -k --since "$started_at_utc" --until "$finished_at_utc" --no-pager \
    >"$run_dir/KERNEL-JOURNAL.txt" 2>&1 || true
  forbidden_tmp=$(mktemp "${TMPDIR:-/tmp}/q38-platform-v3-forbidden.XXXXXX")
  set +e
  grep -REni --include='*.log' --include='*.txt' \
    'ProviderError|illegal memory access|illegal instruction|torch\.AcceleratorError|CUDA error|CUBLAS_STATUS|RPC call .*timed out|RPC.*timeout|EngineCore.*(failed|died)|Engine core proc failed|inference exited or became unhealthy|NVRM: Xid|Xid \([0-9]+\)' \
    "$run_dir" >"$forbidden_tmp"
  forbidden_status=$?
  set -e
  mv "$forbidden_tmp" "$run_dir/FORBIDDEN-INFRASTRUCTURE-SIGNATURES.txt"
  if [[ $forbidden_status -eq 0 ]]; then
    forbidden_status=1
  elif [[ $forbidden_status -eq 1 ]]; then
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
    printf 'schema=q38-platform-v3a1-a6000-qualification-leg/v1\n'
    printf 'platform_lane=platform-v3a1-a6000x2-vllm026\n'
    printf 'teacher_admission_status=permanently_non_admitted\n'
    printf 'label=%s\n' "$label"
    printf 'expected_traces=%s\n' "$expected_traces"
    printf 'baseline_exit_status=%s\n' "$run_status"
    printf 'trace_validation_status=%s\n' "$validation_status"
    printf 'forbidden_signature_status=%s\n' "$forbidden_status"
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
    printf 'cuda_launch_blocking=unset\n'
    printf 'packed_recurrent_decode_override=unset\n'
    printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
    printf 'proposal_sha256=%s\n' "$(sha256sum "$proposal" | awk '{print $1}')"
    printf 'host_manifest_sha256=%s\n' "$(sha256sum "$host_manifest" | awk '{print $1}')"
  } >"$run_dir/PLATFORM-V3-LEG.txt"
  find "$run_dir" -type f ! -name PLATFORM-V3-SHA256SUMS.txt -print0 \
    | sort -z | xargs -0 sha256sum >"$run_dir/PLATFORM-V3-SHA256SUMS.txt"

  if [[ $run_status -ne 0 || $validation_status -ne 0 \
    || $forbidden_status -ne 0 || $teardown_status -ne 0 ]]; then
    echo "$label failed platform-v3a infrastructure qualification" >&2
    return 1
  fi
}

run_leg \
  fresh-sentinel-3806011 \
  1 \
  scripts/run_qwen38_27b_prime_harness_qualification_v1.sh \
  natural_n1a_local \
  1 \
  3806011
run_leg \
  composite-workload-r1 \
  9 \
  scripts/run_q38_platform_v3_composite_workload_v1.sh \
  natural_n1a_local \
  1 \
  3806011
run_leg \
  composite-workload-r2-after-restart \
  9 \
  scripts/run_q38_platform_v3_composite_workload_v1.sh \
  natural_n1a_local \
  1 \
  3806011

qualification_manifest=$output_root/PLATFORM-V3A1-QUALIFIED.txt
{
  printf 'schema=q38-platform-v3a1-a6000-qualification/v1\n'
  printf 'status=qualified\n'
  printf 'platform_lane=platform-v3a1-a6000x2-vllm026\n'
  printf 'teacher_admission_status=qualification_permanently_non_admitted\n'
  printf 'fresh_server_lifetimes=3\n'
  printf 'sentinel_traces=1\n'
  printf 'composite_workload_traces_per_restart=9\n'
  printf 'vllm_version=%s\n' "$installed_version"
  printf 'teacher_model=%s\n' "$teacher_model"
  printf 'teacher_revision=%s\n' "$teacher_revision"
  printf 'qualified_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
  sha256sum "$output_root"/*/PLATFORM-V3-SHA256SUMS.txt
} >"$qualification_manifest"
sha256sum "$qualification_manifest" >"$qualification_manifest.sha256"
echo "platform-v3a1 A6000 qualification passed"
