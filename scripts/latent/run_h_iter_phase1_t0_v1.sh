#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} != --inner ]]; then
  if [[ (${1:-} == --preflight-only || ${1:-} == --full) && $# -eq 4 ]]; then exec timeout --signal=TERM --kill-after=30s 21600s "$0" --inner "$@"; fi
  echo "usage: $0 --preflight-only|--full EXECUTION_COMMIT FINAL_PLAN_FILE_SHA256 h-iter-phase1-t0-train-calibration-run1" >&2; exit 64
fi
shift
readonly mode=$1 execution_commit=$2 plan_sha=$3 run_id=$4
readonly repo=/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1 project=/home/ubuntu/rlm/prime-rl python_bin=/home/ubuntu/rlm/prime-rl/.venv/bin/python3
readonly output=/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-t0-train-calibration-run1
[[ $execution_commit =~ ^[0-9a-f]{40}$ && $plan_sha =~ ^[0-9a-f]{64}$ && $run_id == h-iter-phase1-t0-train-calibration-run1 ]]
[[ -x $python_bin && -d $repo && ! -e $output && ! -L $output ]]
[[ $(git -C "$repo" rev-parse HEAD) == "$execution_commit" && -z $(git -C "$repo" status --porcelain --untracked-files=all) ]]
[[ $(sha256sum "$project/pyproject.toml"|awk '{print $1}') == 504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656 ]]
[[ $(sha256sum "$project/uv.lock"|awk '{print $1}') == fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5 ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo/src" UV_PROJECT_ENVIRONMENT="$project/.venv" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
if [[ $mode == --preflight-only ]]; then export CUDA_VISIBLE_DEVICES=""; exec "$python_bin" "$repo/scripts/latent/run_h_iter_phase1_t0_v1.py" --repo "$repo" --execution-commit "$execution_commit" --plan-file-sha256 "$plan_sha" --run-id "$run_id" --preflight-only; fi
[[ $(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | tr -d '\r') == "NVIDIA RTX A6000" ]]
readonly free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ $free_mib =~ ^[0-9]+$ && $free_mib -ge 45056 ]]
[[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]') ]]
readonly ram_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
readonly disk_kib=$(df -Pk /home/ubuntu/rlm/outputs | awk 'NR==2 {print $4}')
[[ $ram_kib -ge 67108864 && $disk_kib -ge 16777216 ]]
export CUDA_VISIBLE_DEVICES=0
set +e
"$python_bin" "$repo/scripts/latent/run_h_iter_phase1_t0_v1.py" --repo "$repo" --execution-commit "$execution_commit" --plan-file-sha256 "$plan_sha" --run-id "$run_id" --full
readonly result=$?
set -e
"$python_bin" "$repo/scripts/latent/run_h_iter_phase1_t0_v1.py" --repo "$repo" --execution-commit "$execution_commit" --plan-file-sha256 "$plan_sha" --run-id "$run_id" --validate-terminal
exit "$result"
