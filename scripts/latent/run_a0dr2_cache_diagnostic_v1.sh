#!/usr/bin/env bash
set -euo pipefail

if [[ "${A0DR2_OWNER_APPROVED:-}" != "1" ]]; then
  echo "A0DR2_OWNER_APPROVED=1 is required after root and evaluator review" >&2
  exit 2
fi

repo=${1:?absolute repository path required}
run_id=${2:?exclusive A0DR2 run ID required}
plan="$repo/experiments/qwen35-2b-latent-workspace-v1/a0dr2-cache-diagnostic-plan-v1.json"
bank="$repo/experiments/qwen35-2b-latent-workspace-v1/a0-mechanism-bank-v1.json"
coordinator=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
worker=/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8
output_root=/home/ubuntu/rlm/outputs/latent-a0dr2-cache-diagnostic-v1
output_dir="$output_root/$run_id"
uv_bin=/home/ubuntu/.local/bin/uv
shared_project=/home/ubuntu/rlm/prime-rl
shared_venv="$shared_project/.venv"

if [[ "$repo" != /* || -L "$repo" || ! -d "$repo" ]]; then
  echo "repository must be absolute, existing, and non-symlinked" >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^a0dr-r2-[a-z0-9][a-z0-9._-]{2,60}$ ]]; then
  echo "run ID must use the fresh a0dr-r2- namespace" >&2
  exit 2
fi
if [[ -L "$plan" || -L "$bank" || ! -f "$plan" || ! -f "$bank" ]]; then
  echo "frozen A0DR2 plan or bank is absent or symlinked" >&2
  exit 2
fi
if [[ -L "$coordinator" || -L "$worker" || ! -f "$coordinator/STABLE" || ! -f "$worker/STABLE" ]]; then
  echo "protected e33/H176 checkpoints are incomplete or symlinked" >&2
  exit 2
fi
if [[ ! -x "$uv_bin" || ! -d "$shared_venv" ]]; then
  echo "frozen uv executable or shared environment is absent" >&2
  exit 2
fi
if [[ -L "$output_root" ]]; then
  echo "A0DR2 output root must not be a symlink" >&2
  exit 2
fi
install -d -m 700 "$output_root"
if [[ -e "$output_dir" || -L "$output_dir" ]]; then
  echo "refusing to reuse A0DR2 output namespace: $output_dir" >&2
  exit 2
fi

available_kib=$(df -Pk "$output_root" | awk 'NR==2 {print $4}')
host_memory_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
if (( available_kib < 8388608 || host_memory_kib < 67108864 )); then
  echo "A0DR2 host is below the frozen disk or RAM bound" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)" != "NVIDIA RTX A6000" ]]; then
  echo "A0DR2 cuda:0 is not the frozen NVIDIA RTX A6000" >&2
  exit 2
fi
gpu_memory_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i 0 2>/dev/null || true)
if (( gpu_memory_used_mib > 512 )) || [[ -n "$compute_pids" ]]; then
  echo "A0DR2 cuda:0 is not idle" >&2
  exit 2
fi
if pgrep -af '[v]llm|[p]rime_rl.*(trainer|orchestrator|inference)' >/dev/null; then
  echo "A0DR2 refuses to launch while a model service or PrimeRL runtime is active" >&2
  exit 2
fi

cd "$repo"
execution_commit=$(git rev-parse HEAD)
if [[ ! "$execution_commit" =~ ^[0-9a-f]{40}$ ]] || [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "A0DR2 requires an exact commit and clean worktree" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$shared_venv"
export PYTHONPATH="$repo/src"
timeout 30m "$uv_bin" run --project "$shared_project" --no-sync python \
  "$repo/scripts/latent/run_a0dr2_cache_diagnostic_v1.py" \
  --repo "$repo" \
  --plan "$plan" \
  --bank "$bank" \
  --coordinator "$coordinator" \
  --worker "$worker" \
  --output-dir "$output_dir" \
  --execution-commit "$execution_commit" \
  --owner-approved

output_entries=$(find "$output_dir" -mindepth 1 -maxdepth 1 -printf '%f\n')
output_bytes=$(du -sb "$output_dir" | awk '{print $1}')
if [[ "$output_entries" != "receipt.json" || "$output_bytes" -gt 16777216 ]]; then
  echo "A0DR2 output namespace contains unexpected files or exceeds 16 MiB" >&2
  exit 2
fi
