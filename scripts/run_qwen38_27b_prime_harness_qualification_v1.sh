#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=$root/experiments/qwen38-27b-prime-harness-qualification-v1
template=${QWEN38_QUALIFICATION_CONFIG:-$experiment/qualification-template.toml}
model=${1:-Qwen/Qwen3.8-27B}
label=${2:-untouched-qwen38-27b}
evaluation_root=${QWEN38_QUALIFICATION_OUTPUT_ROOT:-${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/evals/qwen38-27b-prime-harness-qualification-v1}}
output_root=$evaluation_root/$label
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
runtime_python=${EVAL_PYTHON_BIN:-$root/.venv/bin/python}
client_base_url=${EVAL_CLIENT_BASE_URL:-http://127.0.0.1:8100/v1}
client_health_url=${EVAL_CLIENT_HEALTH_URL:-${client_base_url%/v1}/health}
axes_csv=${QWEN38_QUALIFICATION_AXES:-direct,atomic_state,atomic_send,atomic_child_request,natural_n1a,natural_n1b,natural_n1a_local,single,natural_n2}
num_tasks=${QWEN38_QUALIFICATION_NUM_TASKS:-1}
num_rollouts=${QWEN38_QUALIFICATION_NUM_ROLLOUTS:-1}
max_concurrent=${QWEN38_QUALIFICATION_MAX_CONCURRENT:-4}
reasoning_effort=${QUALIFICATION_REASONING_EFFORT:-xhigh}
index_offset=${QWEN38_QUALIFICATION_INDEX_OFFSET:-0}
sampling_seed=${QUALIFICATION_SAMPLING_SEED:-20260819}
master_seed=${QWEN38_QUALIFICATION_MASTER_SEED:-20260819}
sampling_temperature=${QUALIFICATION_SAMPLING_TEMPERATURE:-1.0}
privileged_hint_path=${QUALIFICATION_PRIVILEGED_HINT_PATH:-}
privileged_bootstrap_path=${QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH:-}
interaction_curriculum=${PROCEDURAL_INTERACTION_CURRICULUM:-}
eval_max_address_space_bytes=${QWEN38_QUALIFICATION_EVAL_MAX_ADDRESS_SPACE_BYTES:-0}

cd "$root"
uv_bin=${UV_BIN:-$(command -v uv || true)}
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv executable not found" >&2
  exit 1
fi
if [[ ! -f "$template" ]]; then
  echo "qualification template does not exist: $template" >&2
  exit 1
fi
if [[ ! -x "$runtime_python" ]]; then
  echo "evaluation Python is missing: $runtime_python" >&2
  exit 1
fi
if [[ ! "$num_tasks" =~ ^[1-9][0-9]*$ \
  || ! "$num_rollouts" =~ ^[1-9][0-9]*$ \
  || ! "$max_concurrent" =~ ^[1-9][0-9]*$ ]]; then
  echo "qualification task, rollout, and concurrency counts must be positive integers" >&2
  exit 1
fi
if [[ ! "$index_offset" =~ ^[0-9]+$ ]]; then
  echo "QWEN38_QUALIFICATION_INDEX_OFFSET must be a non-negative integer" >&2
  exit 1
fi
if [[ ! "$sampling_seed" =~ ^[0-9]+$ ]]; then
  echo "QUALIFICATION_SAMPLING_SEED must be a non-negative integer" >&2
  exit 1
fi
if [[ ! "$master_seed" =~ ^[0-9]+$ ]]; then
  echo "QWEN38_QUALIFICATION_MASTER_SEED must be a non-negative integer" >&2
  exit 1
fi
if [[ ! "$sampling_temperature" =~ ^(0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(\.[0-9]+)?)$ ]]; then
  echo "QUALIFICATION_SAMPLING_TEMPERATURE must be a positive decimal" >&2
  exit 1
fi
if [[ ! "$eval_max_address_space_bytes" =~ ^[0-9]+$ ]]; then
  echo "QWEN38_QUALIFICATION_EVAL_MAX_ADDRESS_SPACE_BYTES must be a non-negative integer" >&2
  exit 1
fi
if ((eval_max_address_space_bytes > 0)) && ! command -v prlimit >/dev/null; then
  echo "prlimit is required when an evaluation address-space limit is configured" >&2
  exit 1
fi
if [[ -n "$privileged_hint_path" && ! -f "$privileged_hint_path" ]]; then
  echo "privileged hint artifact does not exist: $privileged_hint_path" >&2
  exit 1
fi
if [[ -n "$privileged_bootstrap_path" && ! -f "$privileged_bootstrap_path" ]]; then
  echo "privileged bootstrap artifact does not exist: $privileged_bootstrap_path" >&2
  exit 1
fi
if [[ -n "$privileged_hint_path" && -n "$privileged_bootstrap_path" ]]; then
  echo "privileged hint and bootstrap artifacts are mutually exclusive" >&2
  exit 1
fi
case "$interaction_curriculum" in
  ""|e0_full_actions|e0b_select_child_value|e0c_natural_child|e0c2_natural_child_no_template|e0c25_inline_evidence|e0c275_inline_location|e0c28_inline_only|e0c29_evidence_available|e0c3_natural_child_minimal|e0d_guided_yield|e0d2_capped_yield|e0d2_capped_yield_exact_child|e0d3_uncapped_yield_exact_child|e0d3_uncapped_yield|e1_root_and_yield|e2_yield_only) ;;
  *) echo "unsupported interaction curriculum phase: $interaction_curriculum" >&2; exit 1 ;;
esac
if [[ -n "$interaction_curriculum" ]]; then
  if [[ "$axes_csv" != natural_n1a ]]; then
    echo "interaction curriculum requires exactly the natural_n1a axis" >&2
    exit 1
  fi
  if [[ -z "$privileged_bootstrap_path" ]]; then
    echo "interaction curriculum requires a train-gen bootstrap artifact" >&2
    exit 1
  fi
fi
if [[ -n "${QWEN38_QUALIFICATION_START_INDEX:-}" && "$index_offset" != 0 ]]; then
  echo "qualification start-index override and index offset are mutually exclusive" >&2
  exit 1
fi
case "$reasoning_effort" in
  low|medium|high|xhigh) ;;
  *) echo "unsupported qualification reasoning effort: $reasoning_effort" >&2; exit 1 ;;
esac
if [[ -e "$output_root/VERSIONS.txt" || -e "$output_root/resolved-configs" ]]; then
  echo "refusing to overwrite existing qualification artifacts: $output_root" >&2
  exit 1
fi
if [[ "$client_base_url" == http://127.0.0.1:* || "$client_base_url" == http://localhost:* ]]; then
  if ! curl -fsS "$client_health_url" >/dev/null; then
    echo "local evaluation endpoint is not healthy: $client_health_url" >&2
    exit 1
  fi
fi

mapfile -t vllm_provenance < <(
  "$runtime_python" - <<'PY'
import importlib.metadata
import json

distribution = importlib.metadata.distribution("vllm")
direct_url = "unknown"
for installed_file in distribution.files or ():
    if str(installed_file).endswith("direct_url.json"):
        payload = json.loads(distribution.locate_file(installed_file).read_text())
        direct_url = payload.get("url", "unknown")
        break
print(distribution.version)
print(direct_url)
PY
)
if [[ ${#vllm_provenance[@]} -ne 2 || -z "${vllm_provenance[0]}" ]]; then
  echo "could not resolve installed vLLM provenance" >&2
  exit 1
fi
vllm_version=${vllm_provenance[0]}
vllm_distribution_url=${vllm_provenance[1]}
uv_lock_sha256=$(sha256sum "$root/uv.lock" | awk '{print $1}')
inference_config=$output_root/inference.toml
if [[ -f "$inference_config" ]]; then
  inference_config_sha256=$(sha256sum "$inference_config" | awk '{print $1}')
else
  inference_config_sha256=external_endpoint_not_recorded
fi

"$root/scripts/build_prime_agent_runtime_image_v1.sh" >/dev/null
for package in subagent_communication_v1 procedural_harness_master_v1; do
  "$uv_bin" pip install --python "$runtime_python" --no-deps --editable \
    "$root/deps/verifiers/environments/$package" >/dev/null
done

mkdir -p "$output_root/resolved-configs"
{
  printf 'prime_rl_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'verifiers_commit=%s\n' "$(git -C deps/verifiers rev-parse HEAD)"
  printf 'prime_agent_version=0.7.2-beta.495.1.97b994c\n'
  printf 'model=%s\n' "$model"
  printf 'model_revision=%s\n' "${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
  printf 'vllm_version=%s\n' "$vllm_version"
  printf 'vllm_distribution_url=%s\n' "$vllm_distribution_url"
  printf 'uv_lock_sha256=%s\n' "$uv_lock_sha256"
  printf 'inference_config_sha256=%s\n' "$inference_config_sha256"
  printf 'axes=%s\n' "$axes_csv"
  printf 'num_tasks=%s\n' "$num_tasks"
  printf 'num_rollouts=%s\n' "$num_rollouts"
  printf 'max_concurrent=%s\n' "$max_concurrent"
  printf 'reasoning_effort=%s\n' "$reasoning_effort"
  printf 'index_offset=%s\n' "$index_offset"
  printf 'sampling_seed=%s\n' "$sampling_seed"
  printf 'master_seed=%s\n' "$master_seed"
  printf 'sampling_temperature=%s\n' "$sampling_temperature"
  printf 'privileged_hint_path=%s\n' "${privileged_hint_path:-none}"
  printf 'privileged_bootstrap_path=%s\n' "${privileged_bootstrap_path:-none}"
  printf 'interaction_curriculum=%s\n' "${interaction_curriculum:-none}"
  if [[ -n "$privileged_hint_path" ]]; then
    sha256sum "$privileged_hint_path"
  fi
  if [[ -n "$privileged_bootstrap_path" ]]; then
    sha256sum "$privileged_bootstrap_path"
  fi
  if [[ -n "${EVAL_LORA_PATH:-}" ]]; then
    printf 'base_model=%s\n' "${EVAL_BASE_MODEL:-}"
    printf 'lora_name=%s\n' "${EVAL_LORA_NAME:-}"
    printf 'lora_path=%s\n' "$EVAL_LORA_PATH"
    printf 'lora_rank=%s\n' "${EVAL_LORA_RANK:-}"
    sha256sum \
      "$EVAL_LORA_PATH/adapter_config.json" \
      "$EVAL_LORA_PATH/adapter_model.safetensors"
  fi
  sha256sum \
    "$template" \
    "$root/deps/verifiers/verifiers/v1/dialects/chat.py" \
    "$root/deps/verifiers/datasets/procedural_harness_master_v1/generate.py" \
    "$root/deps/verifiers/environments/procedural_harness_master_v1/procedural_harness_master_v1/taskset.py" \
    "$root/deps/verifiers/environments/procedural_harness_master_v1/procedural_harness_master_v1/interaction_curriculum.py" \
    "$root/deps/verifiers/environments/procedural_harness_master_v1/procedural_harness_master_v1/natural_yield_scaffold.py" \
    "$root/deps/verifiers/environments/procedural_harness_master_v1/procedural_harness_master_v1/causal_context_boundary.py"
} >"$output_root/VERSIONS.txt"

IFS=, read -ra axes <<<"$axes_csv"
trace_dirs=()
for axis in "${axes[@]}"; do
  case "$axis" in
    direct) split=valid_gen; start_index=0; curriculum=none; family_filter=direct ;;
    single) split=valid_gen; start_index=1; curriculum=none; family_filter=single ;;
    atomic_state) split=train_gen; start_index=3801000; curriculum=$axis; family_filter=none ;;
    atomic_send) split=train_gen; start_index=3802000; curriculum=$axis; family_filter=none ;;
    atomic_child_request) split=train_gen; start_index=3803000; curriculum=$axis; family_filter=none ;;
    natural_n1a) split=train_gen; start_index=3804000; curriculum=$axis; family_filter=none ;;
    natural_n1b) split=train_gen; start_index=3805000; curriculum=$axis; family_filter=none ;;
    natural_n1a_local) split=train_gen; start_index=3806000; curriculum=$axis; family_filter=none ;;
    natural_direct_control) split=train_gen; start_index=3807000; curriculum=$axis; family_filter=none ;;
    natural_n2) split=train_gen; start_index=3808000; curriculum=$axis; family_filter=none ;;
    *) echo "unknown qualification axis: $axis" >&2; exit 1 ;;
  esac
  if [[ -n "${QWEN38_QUALIFICATION_START_INDEX:-}" ]]; then
    if [[ ${#axes[@]} -ne 1 ]]; then
      echo "QWEN38_QUALIFICATION_START_INDEX requires exactly one selected axis" >&2
      exit 1
    fi
    if [[ ! "$QWEN38_QUALIFICATION_START_INDEX" =~ ^[0-9]+$ ]]; then
      echo "QWEN38_QUALIFICATION_START_INDEX must be a non-negative integer" >&2
      exit 1
    fi
    start_index=$QWEN38_QUALIFICATION_START_INDEX
  elif [[ "$index_offset" != 0 ]]; then
    start_index=$((start_index + index_offset))
  fi
  resolved=$output_root/resolved-configs/$axis.toml
  "$runtime_python" - "$template" "$resolved" "$axis" "$split" "$start_index" "$curriculum" "$num_tasks" "$num_rollouts" "$family_filter" "$reasoning_effort" "$sampling_seed" "$sampling_temperature" "$privileged_hint_path" "$privileged_bootstrap_path" "$master_seed" <<'PY'
import json
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
replacements = (
    (r"^num_tasks = [0-9]+$", f"num_tasks = {sys.argv[7]}", "num_tasks"),
    (r"^num_rollouts = [0-9]+$", f"num_rollouts = {sys.argv[8]}", "num_rollouts"),
    (r'^split = "[^"]+"$', f'split = "{sys.argv[4]}"', "split"),
    (r"^count = [0-9]+$", f"count = {sys.argv[7]}", "count"),
    (r"^start_index = [0-9]+$", f"start_index = {sys.argv[5]}", "start_index"),
    (r"^master_seed = [0-9]+$", f"master_seed = {sys.argv[15]}", "master_seed"),
    (
        r'^reasoning_effort = "[^"]+"$',
        f'reasoning_effort = "{sys.argv[10]}"',
        "reasoning_effort",
    ),
    (r"^seed = [0-9]+$", f"seed = {sys.argv[11]}", "sampling seed"),
    (
        r"^temperature = [0-9.]+$",
        f"temperature = {sys.argv[12]}",
        "sampling temperature",
    ),
)
for pattern, replacement, description in replacements:
    source, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"qualification template must contain one {description}")
if sys.argv[6] == "none":
    source, rung_count = re.subn(
        r'^curriculum_rung = "[^"]+"\n', "", source, count=1, flags=re.MULTILINE
    )
    source, payload_count = re.subn(
        r'^private_payload_mode = "[^"]+"\n', "", source, count=1, flags=re.MULTILINE
    )
    if rung_count != 1 or payload_count != 1:
        raise SystemExit("qualification template lacks removable curriculum fields")
    source, family_count = re.subn(
        r"^(master_seed = [0-9]+)$",
        rf'\1\nfamilies = ["{sys.argv[9]}"]',
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if family_count != 1:
        raise SystemExit("qualification template lacks a families field")
else:
    source, count = re.subn(
        r'^curriculum_rung = "[^"]+"$',
        f'curriculum_rung = "{sys.argv[6]}"',
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("qualification template must contain one curriculum_rung")
if sys.argv[13]:
    source, count = re.subn(
        r"^(record_causal_feedback = (?:true|false))$",
        rf"\1\nprivileged_hint_path = {json.dumps(sys.argv[13])}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("qualification template must contain record_causal_feedback")
if sys.argv[14]:
    source, count = re.subn(
        r"^(record_causal_feedback = (?:true|false))$",
        rf"\1\nprivileged_bootstrap_path = {json.dumps(sys.argv[14])}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("qualification template must contain record_causal_feedback")
Path(sys.argv[2]).write_text(source)
PY

  eval_command=(
    "$eval_bin" @ "$resolved"
    --model "$model"
    --client.base-url "$client_base_url"
    --max-concurrent "$max_concurrent"
    --output-dir "$output_root"
    --run.name "$axis"
    --run.dir "$axis"
  )
  if ((eval_max_address_space_bytes > 0)); then
    prlimit --as="$eval_max_address_space_bytes" -- "${eval_command[@]}"
  else
    "${eval_command[@]}"
  fi
  trace_dirs+=("$output_root/$axis")
  "$runtime_python" scripts/summarize_procedural_harness_master_v1.py \
    "$output_root/$axis" --rescore --output "$output_root/$axis/SUMMARY.json" >/dev/null
done

"$runtime_python" scripts/summarize_procedural_harness_master_v1.py \
  "${trace_dirs[@]}" --rescore --output "$output_root/SUMMARY.json"
