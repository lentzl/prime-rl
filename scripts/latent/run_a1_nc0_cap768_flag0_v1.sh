#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^[0-9a-f]{64}$ || $3 != a1-nc0-cap768-flag-isolation-run1 ]]; then
  echo "usage: $0 <exact-execution-commit> <authorized-plan-sha256> <fresh-run-id>" >&2
  exit 64
fi
readonly execution_commit=$1
readonly authorized_plan_sha256=$2
readonly run_id=$3
readonly repo=/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1
readonly experiment="$repo/experiments/qwen35-2b-latent-workspace-v1"
readonly plan="$experiment/a1-nc0-cap768-flag0-plan-v1.json"
readonly coordinator=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
readonly worker=/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8
readonly output_root=/home/ubuntu/rlm/outputs/latent-a1-nc0-cap768-flag-isolation-v1
readonly output_dir="$output_root/$run_id"
readonly uv_bin=/home/ubuntu/.local/bin/uv
readonly shared_project=/home/ubuntu/rlm/prime-rl
readonly shared_venv="$shared_project/.venv"

for asset in "$repo" "$coordinator" "$worker" "$shared_venv"; do [[ -d "$asset" && ! -L "$asset" ]] || exit 2; done
[[ -x "$uv_bin" && -f "$plan" && ! -L "$plan" && ! -L "$output_root" && ! -e "$output_dir" ]] || exit 2
available_kib=$(df -Pk /home/ubuntu/rlm/outputs | awk 'NR==2 {print $4}')
host_memory_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
(( available_kib >= 62914560 && host_memory_kib >= 67108864 )) || exit 2
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
mapfile -t gpu_used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
[[ ${#gpu_names[@]} -eq 2 && "${gpu_names[0]}" == "NVIDIA RTX A6000" && "${gpu_names[1]}" == "NVIDIA RTX A6000" ]] || exit 2
(( gpu_used[0] <= 512 && gpu_used[1] <= 512 )) && [[ -z "$pids" ]] || exit 2
cd "$repo"
[[ "$(git rev-parse HEAD)" == "$execution_commit" && -z "$(git status --porcelain --untracked-files=all)" ]] || exit 2
[[ "$(sha256sum "$plan" | cut -d' ' -f1)" == "$authorized_plan_sha256" ]] || exit 2
install -d -m 700 "$output_root"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$shared_venv" PYTHONPATH="$repo/src"
set +e
timeout --signal=TERM --kill-after=60s 3600s "$uv_bin" run --project "$shared_project" --no-sync python \
  "$repo/scripts/latent/run_a1_nc0_cap768_flag0_v1.py" --repo "$repo" --plan "$plan" \
  --coordinator "$coordinator" --worker "$worker" \
  --train-bank "$experiment/a1-nc0-train-bank-v1.json" \
  --output-dir "$output_dir" --execution-commit "$execution_commit" --owner-approved
status=$?
set -e
[[ -d "$output_dir" && ! -L "$output_dir" ]] || exit 2
(( $(du -sb "$output_dir" | awk '{print $1}') <= 134217728 )) || exit 2
mapfile -t entries < <(find "$output_dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
if (( status == 0 )); then
  [[ "${entries[*]}" == "receipt.json" ]] || exit 2
else
  [[ "${entries[*]}" == "failure.json" ]] || exit 2
fi
exit "$status"
