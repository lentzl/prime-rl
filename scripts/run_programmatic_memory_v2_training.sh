#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lane=${1:?usage: run_programmatic_memory_v2_training.sh sft|sdft [start_model]}
start_model=${2:-Qwen/Qwen3.5-27B}
admission_root=${MEMORY_ADMISSION_ROOT:-/ephemeral/subagent-rung/evals/336-339-qwen35-27b-memory-v2-teacher-admission-base-r3}
admission_report=${MEMORY_ADMISSION_REPORT:-$admission_root/admission.json}

case "$lane" in
  sft)
    config=${MEMORY_TRAIN_CONFIG:-$root/configs/debug/subagent-communication/340-qwen35-27b-memory-v2-plain-sft.toml}
    command=(sft @ "$config" --model.name "$start_model")
    ;;
  sdft)
    config=${MEMORY_TRAIN_CONFIG:-$root/configs/debug/subagent-communication/341-qwen35-27b-memory-v2-sdft.toml}
    "$root/.venv/bin/python" "$root/scripts/assess_programmatic_memory_teacher_admission.py" \
      "$admission_root" --output "$admission_report" --require-pass >/dev/null
    command=(rl @ "$config" --model.name "$start_model")
    ;;
  *)
    echo "unknown lane '$lane'; expected sft or sdft" >&2
    exit 2
    ;;
esac

cd "$root"
export PATH="$root/.venv/bin:$PATH"
if [[ ! -x .venv/bin/sft || ! -x .venv/bin/rl ]]; then
  echo "Prime-RL executables are missing; run scripts/setup_prime_agent_mastery_host.sh" >&2
  exit 1
fi
if [[ "$start_model" = /* && (! -d "$start_model" || ! -f "$start_model/STABLE") ]]; then
  echo "starting checkpoint is not stable: $start_model" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi

set -a
source .env
set +a
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN or HF_KEY is required" >&2
  exit 1
fi

"${command[@]}"
