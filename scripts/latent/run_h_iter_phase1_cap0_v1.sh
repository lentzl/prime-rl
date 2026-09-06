#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != "--inner" ]]; then
  if [[ ${1:-} == "--preflight-only" && $# -eq 4 ]]; then
    exec timeout --signal=TERM --kill-after=30s 600s "$0" --inner "$@"
  elif [[ ${1:-} == "--full" && $# -eq 5 ]]; then
    exec timeout --signal=TERM --kill-after=60s 3600s "$0" --inner "$@"
  fi
  echo "usage: $0 --preflight-only EXECUTION_COMMIT PLAN_FILE_SHA256 RUN_ID | --full EXECUTION_COMMIT PLAN_FILE_SHA256 RUN_ID 0" >&2
  exit 64
fi
shift

readonly mode=${1:-}
readonly execution_commit=${2:-}
readonly plan_sha=${3:-}
readonly run_id=${4:-}
readonly repo=/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1
readonly project=/home/ubuntu/rlm/prime-rl
readonly python_bin=/home/ubuntu/rlm/prime-rl/.venv/bin/python3
readonly plan_rel=experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-cap0-v1/cap0-plan.json
readonly sidecar_rel=experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-cap0-v1/cap0-plan.sha256
readonly output=/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-cap0-capture-run1

if [[ ! $execution_commit =~ ^[0-9a-f]{40}$ || ! $plan_sha =~ ^[0-9a-f]{64}$ || $run_id != h-iter-phase1-cap0-capture-run1 ]]; then
  echo "CAP0 launch identity rejected" >&2
  exit 65
fi
[[ -x $python_bin && -d $repo && -d $project/.venv ]]
[[ $(git -C "$repo" rev-parse HEAD) == "$execution_commit" ]]
[[ -z $(git -C "$repo" status --porcelain --untracked-files=all) ]]
[[ $(sha256sum "$repo/$plan_rel" | awk '{print $1}') == "$plan_sha" ]]
[[ $(tr -d '\n' < "$repo/$sidecar_rel") == "$plan_sha" ]]
[[ $(sha256sum "$project/pyproject.toml" | awk '{print $1}') == 504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656 ]]
[[ $(sha256sum "$project/uv.lock" | awk '{print $1}') == fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5 ]]
[[ ! -e $output && ! -L $output ]]

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src" UV_PROJECT_ENVIRONMENT="$project/.venv"

if [[ $mode == "--preflight-only" && $# -eq 4 ]]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$python_bin" "$repo/scripts/latent/run_h_iter_phase1_cap0_v1.py" \
    --repo "$repo" --execution-commit "$execution_commit" --plan-file-sha256 "$plan_sha" \
    --run-id "$run_id" --preflight-only
fi

if [[ $mode != "--full" || $# -ne 5 || ${5:-} != 0 ]]; then
  echo "CAP0 inner launch arguments rejected" >&2
  exit 64
fi
readonly gpu=${5}
[[ $(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader | tr -d '\r') == "NVIDIA RTX A6000" ]]
readonly free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ $free_mib =~ ^[0-9]+$ && $free_mib -ge 45056 ]]
[[ -z $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]') ]]
readonly ram_kib=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
readonly disk_kib=$(df -Pk /home/ubuntu/rlm/outputs | awk 'NR==2 {print $4}')
[[ $ram_kib -ge 67108864 && $disk_kib -ge 16777216 ]]
export CUDA_VISIBLE_DEVICES=0

set +e
"$python_bin" "$repo/scripts/latent/run_h_iter_phase1_cap0_v1.py" \
  --repo "$repo" --execution-commit "$execution_commit" --plan-file-sha256 "$plan_sha" \
  --run-id "$run_id" --gpu "$gpu" --output-dir "$output" --full
readonly proof_exit=$?
set -e
"$python_bin" "$repo/scripts/latent/run_h_iter_phase1_cap0_v1.py" \
  --repo "$repo" --execution-commit "$execution_commit" --plan-file-sha256 "$plan_sha" \
  --run-id "$run_id" --output-dir "$output" --validate-terminal
if [[ $proof_exit -eq 0 ]]; then
  [[ -f $output/CAP0-PROOF.json && ! -e $output/CAP0-FAILURE.json ]]
  exit 0
fi
[[ -f $output/CAP0-FAILURE.json && ! -e $output/CAP0-PROOF.json ]]
exit "$proof_exit"
