#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: $0 <exact-clean-execution-commit> <root-authorized-plan-file-sha256> <fresh-run-id>" >&2
  exit 64
fi
readonly execution_commit=$1
readonly authorized_plan_sha256=$2
run_id=$3
[[ "$run_id" =~ ^a1-nc0-r1-[a-z0-9][a-z0-9._-]{2,60}$ ]] || exit 2
readonly repo=/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1
[[ -d "$repo" && ! -L "$repo" ]] || exit 2

experiment="$repo/experiments/qwen35-2b-latent-workspace-v1"
plan="$experiment/a1-nc0-r1-plan-v1.json"
schedule="$experiment/a1-nc0-schedule-v1.json"
disjointness="$experiment/a1-nc0-disjointness-v1.json"
train_bank="$experiment/a1-nc0-train-bank-v1.json"
validation_bank="$experiment/a1-nc0-validation-bank-v1.json"
held_out_bank="$experiment/a1-nc0-held_out-bank-v1.json"
coordinator=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
worker=/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8
output_root=/home/ubuntu/rlm/outputs/latent-a1-nc0-r1-nomination-v1
output_dir="$output_root/$run_id"
uv_bin=/home/ubuntu/.local/bin/uv
shared_project=/home/ubuntu/rlm/prime-rl
shared_venv="$shared_project/.venv"

for asset in "$plan" "$schedule" "$disjointness" "$train_bank" "$validation_bank" "$held_out_bank"; do
  [[ -f "$asset" && ! -L "$asset" ]] || exit 2
done
[[ -f "$coordinator/STABLE" && ! -L "$coordinator" && -f "$worker/STABLE" && ! -L "$worker" ]] || exit 2
[[ -x "$uv_bin" && -d "$shared_venv" && ! -L "$shared_venv" ]] || exit 2
[[ ! -L "$output_root" ]] || exit 2
[[ ! -e "$output_dir" && ! -L "$output_dir" ]] || exit 2
available_kib=$(df -Pk /home/ubuntu/rlm/outputs | awk 'NR==2 {print $4}')
host_memory_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
(( available_kib >= 62914560 && host_memory_kib >= 67108864 )) || exit 2
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#gpu_names[@]} -eq 2 && "${gpu_names[0]}" == "NVIDIA RTX A6000" && "${gpu_names[1]}" == "NVIDIA RTX A6000" ]] || exit 2
mapfile -t gpu_used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
(( gpu_used[0] <= 512 && gpu_used[1] <= 512 )) && [[ -z "$pids" ]] || exit 2
! pgrep -af '[v]llm|[p]rime_rl.*(trainer|orchestrator|inference)' >/dev/null || exit 2

cd "$repo"
commit=$(git rev-parse HEAD)
[[ "$commit" == "$execution_commit" && -z "$(git status --porcelain --untracked-files=all)" ]] || exit 2
[[ "$(sha256sum "$plan" | cut -d' ' -f1)" == "$authorized_plan_sha256" ]] || exit 2
install -d -m 700 "$output_root"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$shared_venv" PYTHONPATH="$repo/src"
set +e
timeout --signal=TERM --kill-after=60s 8h "$uv_bin" run --project "$shared_project" --no-sync python \
  "$repo/scripts/latent/run_a1_nc0_nomination_v1.py" \
  --repo "$repo" --plan "$plan" --schedule "$schedule" --disjointness "$disjointness" \
  --train-bank "$train_bank" --validation-bank "$validation_bank" --held-out-bank "$held_out_bank" \
  --coordinator "$coordinator" --worker "$worker" --output-dir "$output_dir" \
  --execution-commit "$execution_commit" --authorized-plan-sha256 "$authorized_plan_sha256" --owner-approved
run_status=$?
set -e
[[ -d "$output_dir" && ! -L "$output_dir" ]] || exit 2
bytes=$(du -sb "$output_dir" | awk '{print $1}')
(( bytes <= 1073741824 )) || exit 2
mapfile -t entries < <(find "$output_dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
if (( run_status == 0 )); then
  [[ "${entries[*]}" == "bridge-candidate.safetensors receipt.json" ]] || exit 2
else
  [[ "${entries[*]}" == "failure.json" || "${entries[*]}" == "bridge-candidate.safetensors failure.json" ]] || exit 2
fi
exit "$run_status"
