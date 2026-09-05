#!/usr/bin/env bash
set -euo pipefail
[[ "${A0NC_OWNER_APPROVED:-}" == "1" ]] || { echo "A0NC_OWNER_APPROVED=1 required" >&2; exit 2; }
repo=${1:?absolute repo required}
run_id=${2:?exclusive run id required}
[[ "$repo" == /* && -d "$repo" && ! -L "$repo" ]] || exit 2
[[ "$run_id" =~ ^a0dr-nc-[a-z0-9][a-z0-9._-]{2,60}$ ]] || exit 2
plan="$repo/experiments/qwen35-2b-latent-workspace-v1/a0-nocache-plan-v1.json"
bank="$repo/experiments/qwen35-2b-latent-workspace-v1/a0-nocache-bank-v1.json"
coordinator=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
worker=/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8
output_root=/home/ubuntu/rlm/outputs/latent-a0-nocache-receiver-v1
output_dir="$output_root/$run_id"
uv_bin=/home/ubuntu/.local/bin/uv
shared_project=/home/ubuntu/rlm/prime-rl
shared_venv="$shared_project/.venv"
[[ -f "$plan" && ! -L "$plan" && -f "$bank" && ! -L "$bank" ]] || exit 2
[[ -f "$coordinator/STABLE" && ! -L "$coordinator" && -f "$worker/STABLE" && ! -L "$worker" ]] || exit 2
[[ -x "$uv_bin" && -d "$shared_venv" ]] || exit 2
[[ ! -L "$output_root" ]] || exit 2
install -d -m 700 "$output_root"
[[ ! -e "$output_dir" && ! -L "$output_dir" ]] || exit 2
available_kib=$(df -Pk "$output_root" | awk 'NR==2 {print $4}')
host_memory_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
(( available_kib >= 8388608 && host_memory_kib >= 67108864 )) || exit 2
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)" == "NVIDIA RTX A6000" ]] || exit 2
gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i 0 2>/dev/null || true)
(( gpu_used <= 512 )) && [[ -z "$pids" ]] || exit 2
! pgrep -af '[v]llm|[p]rime_rl.*(trainer|orchestrator|inference)' >/dev/null || exit 2
cd "$repo"
commit=$(git rev-parse HEAD)
[[ "$commit" =~ ^[0-9a-f]{40}$ && -z "$(git status --porcelain --untracked-files=all)" ]] || exit 2
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$shared_venv" PYTHONPATH="$repo/src"
timeout 30m "$uv_bin" run --project "$shared_project" --no-sync python \
  "$repo/scripts/latent/run_a0_nocache_receiver_v1.py" \
  --repo "$repo" --plan "$plan" --bank "$bank" --coordinator "$coordinator" --worker "$worker" \
  --output-dir "$output_dir" --execution-commit "$commit" --owner-approved
entries=$(find "$output_dir" -mindepth 1 -maxdepth 1 -printf '%f\n')
bytes=$(du -sb "$output_dir" | awk '{print $1}')
[[ "$entries" == "receipt.json" && "$bytes" -le 16777216 ]] || exit 2
