#!/usr/bin/env bash
set -euo pipefail

if [[ ${1-} != --inner ]]; then
  exec timeout --signal=TERM --kill-after=60s 1800s "$0" --inner "$@"
fi
shift
[[ $# -eq 3 ]] || exit 64
[[ $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[0-9a-f]{64}$ ]] || exit 64
[[ $3 == h-iter-phase1-train-calibration-prereg-run1 ]] || exit 64

readonly execution_commit=$1
readonly plan_file_sha256=$2
readonly run_identity=$3
readonly repo=/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1
readonly shared_project=/home/ubuntu/rlm/prime-rl
readonly shared_venv=/home/ubuntu/rlm/prime-rl/.venv
readonly shared_python=/home/ubuntu/rlm/prime-rl/.venv/bin/python3
readonly uv_bin=/home/ubuntu/.local/bin/uv
readonly plan="$repo/experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1/mf0-plan.json"
readonly sidecar="$repo/experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-train-calibration-v1/mf0-plan.sha256"
readonly runner="$repo/scripts/latent/run_h_iter_phase1_mf0_v1.py"
readonly output=/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-train-calibration-prereg-run1

[[ $output == /home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-train-calibration-prereg-run1 ]] || exit 2
[[ ! -e $output && ! -L $output ]] || exit 2
[[ -x $uv_bin && -x $shared_python && -d $repo && ! -L $repo ]] || exit 2
[[ -f $plan && ! -L $plan && -f $sidecar && ! -L $sidecar ]] || exit 2
[[ $(sha256sum "$shared_project/pyproject.toml" | cut -d' ' -f1) == 504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656 ]] || exit 2
[[ $(sha256sum "$shared_project/uv.lock" | cut -d' ' -f1) == fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5 ]] || exit 2
[[ $(sha256sum "$plan" | cut -d' ' -f1) == "$plan_file_sha256" ]] || exit 2
[[ $(<"$sidecar") == "$plan_file_sha256" ]] || exit 2
[[ $(df -Pk "$repo" | awk 'NR==2 {print $4}') -ge 8388608 ]] || exit 2
[[ $(awk '/MemTotal:/ {print $2}' /proc/meminfo) -ge 8388608 ]] || exit 2
cd "$repo"
[[ $(git rev-parse HEAD) == "$execution_commit" ]] || exit 2
[[ -z $(git status --porcelain --untracked-files=all) ]] || exit 2

export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src"
export UV_PROJECT_ENVIRONMENT="$shared_venv"

set +e
"$uv_bin" run --project "$shared_project" --no-sync "$shared_python" "$runner" \
  --repo "$repo" --plan "$plan" --plan-file-sha256 "$plan_file_sha256" \
  --execution-commit "$execution_commit" --output-dir "$output"
readonly run_status=$?
set -e

"$uv_bin" run --project "$shared_project" --no-sync "$shared_python" "$runner" \
  --repo "$repo" --plan "$plan" --plan-file-sha256 "$plan_file_sha256" \
  --execution-commit "$execution_commit" --output-dir "$output" --validate-terminal

if [[ $run_status -eq 0 ]]; then
  [[ -f "$output/MF0-PROOF.json" && ! -L "$output/MF0-PROOF.json" ]] || exit 2
else
  [[ -f "$output/MF0-FAILURE.json" && ! -L "$output/MF0-FAILURE.json" ]] || exit 2
fi
exit "$run_status"
