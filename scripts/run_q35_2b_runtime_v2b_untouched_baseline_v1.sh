#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_root=${RUNTIME_V2_ROOT:-/home/ubuntu/rlm/runtime-v2}
runtime_env=${RUNTIME_V2_ENV:-$runtime_root/envs/prime-rl-vllm-0.27.1-cu129}
qualification_root=${RUNTIME_V2B_QUALIFICATION_ROOT:-/home/ubuntu/rlm/results/q38-runtime-v2b-blocking-qualification-v1}
output_root=${Q35_2B_V2B_BASELINE_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-runtime-v2b-baseline-v1}
student_snapshot=${Q35_2B_STUDENT_SNAPSHOT:-/home/ubuntu/rlm/models/qwen35-2b-orchestrator-candidate-r1-c90f5a7}
student_revision=local
student_weight_sha256=c75915dd41cd4fc9b1a1ef5582c6fd14913fc6f9971a58feca3b72b4bfcad406
qualification_manifest=$qualification_root/RUNTIME-V2B-QUALIFIED.txt
qualification_manifest_sha256=d827b6a1a9fc25b92bd79e46d89e3fb82737c84a34ed77a76ce22e134e9e1992
label=untouched-q35-2b-runtime-v2b-principal16-r1
run_dir=$output_root/$label
wrapper_path=$root/scripts/run_q35_2b_runtime_v2b_untouched_baseline_v1.sh
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
if [[ ! -f "$student_snapshot/STABLE" \
  || ! -f "$student_snapshot/model.safetensors" ]]; then
  echo "student snapshot is not a stable dense export: $student_snapshot" >&2
  exit 1
fi
actual_student_sha256=$(sha256sum "$student_snapshot/model.safetensors" | awk '{print $1}')
if [[ "$actual_student_sha256" != "$student_weight_sha256" ]]; then
  echo "student weight hash mismatch: $actual_student_sha256" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing untouched baseline while a GPU process is active" >&2
  exit 1
fi
if [[ -e "$output_root" ]]; then
  echo "refusing to append to or overwrite untouched baseline: $output_root" >&2
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
  PRIME_MASTERY_OUTPUT_ROOT="$output_root" \
  "$root/scripts/run_q35_2b_role_distillation_eval_v1.sh" \
  "$student_snapshot" "$label" "$student_revision"
run_status=$?
set -e
finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [[ ! -d "$run_dir" ]]; then
  echo "untouched baseline produced no run directory: $run_dir" >&2
  exit 1
fi

set +e
"$runtime_env/bin/python" - "$run_dir" "$student_snapshot" \
  >"$run_dir/BASELINE-AUDIT.json" \
  2>"$run_dir/BASELINE-AUDIT.stderr" <<'PY'
import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
student = pathlib.Path(sys.argv[2]).resolve()
axes = {
    "natural_direct_control": 3907000,
    "natural_n1a": 3904000,
    "natural_n1a_local": 3906000,
    "natural_n1b": 3905000,
}
result = {}
all_trace_ids = set()
all_task_keys = set()
for axis, start_index in axes.items():
    trace_path = run_dir / axis / "traces.jsonl"
    if not trace_path.is_file():
        raise SystemExit(f"missing baseline traces for {axis}")
    envelopes = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(envelopes) != 16:
        raise SystemExit(f"axis {axis} has {len(envelopes)} envelopes; expected 16")
    traces = []
    for envelope in envelopes:
        if envelope.get("ok") is not True or envelope.get("errors"):
            raise SystemExit(f"axis {axis} contains a runtime-error envelope")
        if len(envelope.get("traces") or []) != 1:
            raise SystemExit(f"axis {axis} envelope does not contain exactly one trace")
        traces.append(envelope["traces"][0])
    indices = []
    hard_successes = 0
    for trace in traces:
        if trace.get("ok") is not True or trace.get("errors"):
            raise SystemExit(f"axis {axis} contains a runtime-error trace")
        if trace.get("is_completed") is not True:
            raise SystemExit(f"axis {axis} contains an incomplete trace")
        model = (((trace.get("agent") or {}).get("config") or {}).get("model"))
        if not isinstance(model, str) or pathlib.Path(model).resolve() != student:
            raise SystemExit(f"axis {axis} trace has the wrong student model: {model}")
        trace_id = trace.get("id")
        task_key = (trace.get("task") or {}).get("key")
        if not isinstance(trace_id, str) or not isinstance(task_key, str):
            raise SystemExit(f"axis {axis} trace lacks identity")
        if trace_id in all_trace_ids or task_key in all_task_keys:
            raise SystemExit(f"duplicate baseline identity: {trace_id}/{task_key}")
        all_trace_ids.add(trace_id)
        all_task_keys.add(task_key)
        match = re.search(r"-(\d{8})-", task_key)
        if match is None or f"-{axis}-" not in task_key:
            raise SystemExit(f"unexpected baseline task key: {task_key}")
        indices.append(int(match.group(1)))
        reward = (trace.get("rewards") or {}).get("harness_score") or {}
        score = reward.get("score", reward.get("value", 0.0))
        hard_successes += float(score or 0.0) == 1.0
    expected_indices = list(range(start_index, start_index + 16))
    if sorted(indices) != expected_indices:
        raise SystemExit(f"unexpected {axis} task indices: {sorted(indices)}")
    result[axis] = {
        "complete_traces": len(traces),
        "hard_successes": hard_successes,
        "start_index": start_index,
    }
print(json.dumps(result, indent=2, sort_keys=True))
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
  printf 'schema=q35-2b-runtime-v2b-untouched-baseline-run/v1\n'
  printf 'runtime_lane=runtime-v2b-blocking\n'
  printf 'student_state=untouched_immutable_dense_snapshot\n'
  printf 'cuda_launch_blocking=1\n'
  printf 'baseline_exit_status=%s\n' "$run_status"
  printf 'baseline_audit_status=%s\n' "$audit_status"
  printf 'forbidden_signature_status=%s\n' "$forbidden_status"
  printf 'teardown_status=%s\n' "$teardown_status"
  printf 'started_at_utc=%s\n' "$started_at_utc"
  printf 'finished_at_utc=%s\n' "$finished_at_utc"
  printf 'vllm_version=%s\n' "$installed_version"
  printf 'student_snapshot=%s\n' "$student_snapshot"
  printf 'student_weight_sha256=%s\n' "$student_weight_sha256"
  printf 'runtime_qualification_manifest_sha256=%s\n' "$qualification_manifest_sha256"
  printf 'wrapper_sha256=%s\n' "$wrapper_sha256"
} >"$run_dir/RUNTIME-V2B-BASELINE-RUN.txt"

if [[ $run_status -ne 0 || $audit_status -ne 0 \
  || $forbidden_status -ne 0 || $teardown_status -ne 0 ]]; then
  find "$run_dir" -type f ! -name RUNTIME-V2B-BASELINE-SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"$run_dir/RUNTIME-V2B-BASELINE-SHA256SUMS.txt"
  echo "untouched 2B baseline failed infrastructure validation" >&2
  exit 1
fi

"$runtime_env/bin/python" "$root/scripts/build_q35_2b_baseline_manifest_v1.py" \
  --baseline-run "$run_dir" \
  --student-snapshot "$student_snapshot" \
  --runtime-qualification "$qualification_manifest" \
  --output "$output_root/BASELINE-MANIFEST.json"

find "$run_dir" -type f ! -name RUNTIME-V2B-BASELINE-SHA256SUMS.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"$run_dir/RUNTIME-V2B-BASELINE-SHA256SUMS.txt"
sha256sum "$output_root/BASELINE-MANIFEST.json" \
  >"$output_root/BASELINE-MANIFEST.json.sha256"
echo "untouched Qwen3.5-2B runtime-v2b baseline completed"
