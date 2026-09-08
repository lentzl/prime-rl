#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:?model name required}
label=${2:?evaluation label required}
config=${DOCUMENT_SUMMARY_CONFIG:-$root/experiments/qwen35-2b-document-summary-prime-agent-v1/smoke.toml}
output_root=${QWEN38_QUALIFICATION_OUTPUT_ROOT:-/home/ubuntu/rlm/results/q35-2b-document-summary-prime-agent-v1}
run_output=$output_root/$label/document
eval_bin=${EVAL_BIN:-$root/.venv/bin/eval}
runtime_python=${EVAL_PYTHON_BIN:-$root/.venv/bin/python}
uv_bin=${UV_BIN:-$(command -v uv || true)}

if [[ ! -x "$eval_bin" && -x /home/ubuntu/rlm/prime-rl/.venv/bin/eval ]]; then
  eval_bin=/home/ubuntu/rlm/prime-rl/.venv/bin/eval
fi
if [[ ! -x "$runtime_python" && -x /home/ubuntu/rlm/prime-rl/.venv/bin/python ]]; then
  runtime_python=/home/ubuntu/rlm/prime-rl/.venv/bin/python
fi
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin=$HOME/.local/bin/uv
fi
if [[ -z "$uv_bin" || ! -x "$eval_bin" || ! -x "$runtime_python" ]]; then
  echo "document summary evaluation runtime is incomplete" >&2
  exit 1
fi
if [[ ! -f "$config" ]]; then
  echo "document summary config not found: $config" >&2
  exit 1
fi

cd "$root"
export PYTHONPATH="$root/deps/verifiers/environments/document_summary_v1:$root/deps/verifiers${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$run_output"
{
  printf 'prime_rl_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'verifiers_commit=%s\n' "$(git -C deps/verifiers rev-parse HEAD)"
  printf 'model=%s\n' "$model"
  printf 'model_revision=%s\n' "${MODEL_REVISION:-candidate-local}"
  sha256sum "$config"
} >"$run_output/VERSIONS.txt"

"$eval_bin" @ "$config" \
  --model "$model" \
  --client.base-url "${EVAL_CLIENT_BASE_URL:?EVAL_CLIENT_BASE_URL is required}" \
  --output-dir "$run_output" \
  --run.name document-summary \
  --run.dir document-summary

"$uv_bin" run --no-sync scripts/summarize_prime_agent_mastery_v2.py \
  "$run_output/document-summary" \
  --expected-count 1 \
  --json >"$run_output/SUMMARY.json"

trace_path=$run_output/document-summary/traces.jsonl
for artifact in source notes extracted_notes summary; do
  key=evidence_$artifact
  selector='.traces[] | select(.task.type == "DocumentSummaryEvidenceTask") | .info[$key] | select(type == "string" and length > 0)'
  if jq -e --arg key "$key" "$selector" "$trace_path" >/dev/null; then
    mkdir -p "$run_output/artifacts"
    jq -j --arg key "$key" "$selector" "$trace_path" >"$run_output/artifacts/$artifact.md"
  fi
done

owner_selector='.traces[] | select(.task.type == "DocumentSummaryMarkdownTask")'
if jq -e "$owner_selector" "$trace_path" >/dev/null; then
  mkdir -p "$run_output/artifacts/chapters"
  jq "$owner_selector | {jobs: .task.data.jobs, receipts: .info.chapter_receipts, errors: .info.summary_file_errors}" \
    "$trace_path" >"$run_output/artifacts/handoffs.json"
  selector="$owner_selector | .info.document_summary_markdown | select(type == \"string\")"
  if jq -e "$selector" "$trace_path" >/dev/null; then
    jq -j "$selector" "$trace_path" >"$run_output/artifacts/summary.md"
  fi
  while IFS= read -r index; do
    for artifact in source summary; do
      selector="$owner_selector | .task.data.document.chapters[\$index].id as \$chapter | "
      if [[ "$artifact" == source ]]; then
        selector+='.info.chapter_sources[$chapter]'
      else
        selector+='. as $trace | .task.data.jobs | to_entries[] | select(.value.chapter_id == $chapter) | $trace.info.chapter_summary_files[.key]'
      fi
      selector+=' | select(type == "string")'
      if jq -e --argjson index "$index" "$selector" "$trace_path" >/dev/null; then
        jq -j --argjson index "$index" "$selector" "$trace_path" \
          >"$run_output/artifacts/chapters/$index-$artifact.md"
      fi
    done
  done < <(jq -r "$owner_selector | .task.data.document.chapters | keys[]" "$trace_path")
fi
