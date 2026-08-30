#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
coordinator_model=${1:?coordinator model path required}
child_model=${2:?child model path required}
batch_dir=${3:?batch artifact directory required}
results_root=${4:?results root required}
batch_id=${5:?batch id required}
revision=${6:?model revision required}
track=${7:?interaction track required}
phase=${8:?interaction phase required}
start_index=${9:?start index required}
tasks=${10:?task count required}
memory=${11:?environment memory path required}
external_model=${12:?external model name required}
designer_role=${13:-coordinator}
document_corpus=${SPADE_DESIGNER_DOCUMENT_CORPUS:-$(dirname "$memory")/prime-agent-designer-docs-v1.json}
inference_bin=${INFERENCE_BIN:-$root/.venv/bin/inference}
uv_bin=${UV_BIN:-/home/ubuntu/.local/bin/uv}
coordinator_port=${COORDINATOR_BACKEND_PORT:-8101}
child_port=${CHILD_BACKEND_PORT:-8102}
proxy_port=${DUAL_PROXY_PORT:-8100}

cd "$root"
if [[ ! -x "$uv_bin" ]]; then
  echo "uv executable is unavailable: $uv_bin" >&2
  exit 1
fi
for model in "$coordinator_model" "$child_model"; do
  if [[ "$model" != /* || ! -f "$model/STABLE" || ! -f "$model/model.safetensors" ]]; then
    echo "dense role model is not an absolute complete checkpoint: $model" >&2
    exit 1
  fi
done
if [[ -e "$batch_dir" ]]; then
    echo "refusing to overwrite coevolution batch: $batch_dir" >&2
    exit 1
fi
if [[ ! -f "$document_corpus" ]]; then
  echo "Environment Designer document corpus is unavailable: $document_corpus" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch coevolution while a GPU process is active" >&2
  exit 1
fi
mkdir -p "$batch_dir"
generation_dir=$batch_dir/generation

write_inference_config() {
  local path=$1 model=$2 backend_port=$3 router_port=$4 rpc_port=$5
  cat >"$path" <<EOF
backend_port = $backend_port

[server]
port = $router_port
liveness_timeout_seconds = 30.0

[vllm]
model = "$model"
revision = "$revision"
dtype = "bfloat16"
max_model_len = 32768
language_model_only = true
enforce_eager = true
trust_remote_code = false
tool_call_parser = "qwen3_coder"
reasoning_parser = "qwen3"
tensor_parallel_size = 1
disable_custom_all_reduce = false
data_parallel_size = 1
data_parallel_rpc_port = $rpc_port
gpu_memory_utilization = 0.80
max_num_seqs = 8
max_num_batched_tokens = 4096

[log]
level = "info"
vf_level = "info"
json_logging = false
log_data = false
interval = 10.0
EOF
}

coordinator_config=$batch_dir/coordinator-inference.toml
child_config=$batch_dir/child-inference.toml
write_inference_config "$coordinator_config" "$coordinator_model" "$coordinator_port" 8001 13346
write_inference_config "$child_config" "$child_model" "$child_port" 8002 13347

coordinator_pid=
child_pid=
proxy_pid=
cleanup() {
  trap - EXIT INT TERM
  for pid in "$proxy_pid" "$child_pid" "$coordinator_pid"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 "$inference_bin" @ "$coordinator_config" >"$batch_dir/coordinator-inference.log" 2>&1 &
coordinator_pid=$!
CUDA_VISIBLE_DEVICES=1 "$inference_bin" @ "$child_config" >"$batch_dir/child-inference.log" 2>&1 &
child_pid=$!
for _ in $(seq 1 480); do
  if ! kill -0 "$coordinator_pid" 2>/dev/null || ! kill -0 "$child_pid" 2>/dev/null; then
    echo "a role inference process exited during startup" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$coordinator_port/health" >/dev/null \
    && curl -fsS "http://127.0.0.1:$child_port/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$coordinator_port/health" >/dev/null
curl -fsS "http://127.0.0.1:$child_port/health" >/dev/null

coordinator_sha=$(sha256sum "$coordinator_model/model.safetensors" | awk '{print $1}')
child_sha=$(sha256sum "$child_model/model.safetensors" | awk '{print $1}')
case "$designer_role" in
  coordinator)
    designer_port=$coordinator_port
    designer_model=$coordinator_model
    designer_sha=$coordinator_sha
    ;;
  child)
    designer_port=$child_port
    designer_model=$child_model
    designer_sha=$child_sha
    ;;
  *)
    echo "unsupported Environment Designer role: $designer_role" >&2
    exit 1
    ;;
esac
"$uv_bin" run --no-sync scripts/q35_2b_spade_coevolution_v1.py generate \
  --base-url "http://127.0.0.1:$designer_port/v1" \
  --model "$designer_model" \
  --model-sha256 "$designer_sha" \
  --designer-role "$designer_role" \
  --batch-id "$batch_id" \
  --track "$track" \
  --phase "$phase" \
  --start-index "$start_index" \
  --tasks "$tasks" \
  --memory "$memory" \
  --document-corpus "$document_corpus" \
  --documents-per-candidate 3 \
  --output-dir "$generation_dir"

if [[ -f "$generation_dir/REJECTIONS.json" && ! -f "$generation_dir/GENERATION.json" ]]; then
  printf 'designer_rejected\n' >"$batch_dir/DESIGNER_REJECTED"
  exit 0
fi

run_arm() {
  local arm=$1 bootstrap=$2
  local label="$batch_id-$arm"
  local run_output="$results_root/$label"
  local routing="$run_output/ROUTING_AUDIT.jsonl"
  if [[ -e "$run_output" ]]; then
    echo "refusing to overwrite coevolution arm: $run_output" >&2
    exit 1
  fi
  mkdir -p "$run_output"
  "$uv_bin" run --no-sync scripts/dual_policy_openai_proxy_v1.py \
    --port "$proxy_port" \
    --coordinator-url "http://127.0.0.1:$coordinator_port/v1" \
    --coordinator-model "$coordinator_model" \
    --child-url "http://127.0.0.1:$child_port/v1" \
    --child-model "$child_model" \
    --external-model "$external_model" \
    --audit-log "$routing" >"$run_output/proxy.log" 2>&1 &
  proxy_pid=$!
  for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$proxy_port/health" >/dev/null && break
    kill -0 "$proxy_pid" 2>/dev/null || exit 1
    sleep 1
  done
  curl -fsS "http://127.0.0.1:$proxy_port/health" >/dev/null
  MODEL_REVISION="$revision" \
  EVAL_CLIENT_BASE_URL="http://127.0.0.1:$proxy_port/v1" \
  QWEN38_QUALIFICATION_OUTPUT_ROOT="$results_root" \
  QWEN38_QUALIFICATION_AXES=natural_n1a \
  QWEN38_QUALIFICATION_NUM_TASKS="$tasks" \
  QWEN38_QUALIFICATION_NUM_ROLLOUTS=1 \
  QWEN38_QUALIFICATION_MAX_CONCURRENT="$tasks" \
  QWEN38_QUALIFICATION_EVAL_MAX_ADDRESS_SPACE_BYTES=$((32 * 1024 * 1024 * 1024)) \
  QWEN38_QUALIFICATION_START_INDEX="$start_index" \
  QUALIFICATION_REASONING_EFFORT=high \
  QUALIFICATION_SAMPLING_SEED=20260823 \
  QUALIFICATION_SAMPLING_TEMPERATURE=0.6 \
  QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH="$bootstrap" \
  PROCEDURAL_INTERACTION_CURRICULUM="$phase" \
  scripts/run_qwen38_27b_prime_harness_qualification_v1.sh "$external_model" "$label"
  kill "$proxy_pid" 2>/dev/null || true
  wait "$proxy_pid" 2>/dev/null || true
  proxy_pid=
  {
    printf 'spade_coevolution_batch=%s\n' "$batch_id"
    printf 'spade_coevolution_arm=%s\n' "$arm"
    printf 'coordinator_model_sha256=%s\n' "$coordinator_sha"
    printf 'child_model_sha256=%s\n' "$child_sha"
    sha256sum "$routing" "$bootstrap" "$coordinator_config" "$child_config"
  } >>"$run_output/VERSIONS.txt"
}

run_arm no-hint "$generation_dir/NO_HINT_BOOTSTRAP.json"
run_arm hint "$generation_dir/HINT_BOOTSTRAP.json"
for arm in no-hint hint; do
  "$uv_bin" run --no-sync scripts/summarize_q35_2b_interaction_curriculum_v1.py \
    "$results_root/$batch_id-$arm/natural_n1a/traces.jsonl" \
    --phase "$phase" \
    --output "$batch_dir/${arm^^}_SUMMARY.json"
done
"$uv_bin" run --no-sync scripts/q35_2b_spade_coevolution_v1.py score \
  --generation "$generation_dir/GENERATION.json" \
  --no-hint-summary "$batch_dir/NO-HINT_SUMMARY.json" \
  --hint-summary "$batch_dir/HINT_SUMMARY.json" \
  --memory "$memory" \
  --output "$batch_dir/SCORE.json"
"$uv_bin" run --no-sync python - \
  "$batch_dir" "$generation_dir" "$designer_role" "$track" "$phase" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

batch_dir, generation_dir = Path(sys.argv[1]), Path(sys.argv[2])
designer_role, track, phase = sys.argv[3:]

def load(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

counts = {
    "no-hint": len(load(batch_dir / "NO-HINT_SUMMARY.json").get("qualifying") or []),
    "hint": len(load(batch_dir / "HINT_SUMMARY.json").get("qualifying") or []),
}
# Leak while capability is weak; remove the generated hint once the unhinted
# arm independently clears the unchanged four-trajectory promotion floor.
selected = "no-hint" if counts["no-hint"] >= 4 else "hint"
bootstrap = generation_dir / ("NO_HINT_BOOTSTRAP.json" if selected == "no-hint" else "HINT_BOOTSTRAP.json")
score = batch_dir / "SCORE.json"
payload = {
    "schema_version": "qwen35-2b-role-local-designer-selection/v1",
    "designer_role": designer_role,
    "track": track,
    "phase": phase,
    "qualifying_by_arm": counts,
    "selected_arm": selected,
    "selection_rule": "unhinted_after_four_else_leak",
    "bootstrap_path": str(bootstrap.resolve()),
    "bootstrap_sha256": digest(bootstrap),
    "score_path": str(score.resolve()),
    "score_sha256": digest(score),
}
(batch_dir / "TRAINING_SELECTION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
printf 'complete\n' >"$batch_dir/PAIRED_EVALUATIONS_COMPLETE"
