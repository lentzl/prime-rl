#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != --inner ]]; then
  if [[ ${1:-} == --proof && $# -eq 4 ]]; then
    exec timeout --signal=TERM --kill-after=60s 1800s "$0" --inner "$@"
  fi
  echo "usage: $0 --proof EXECUTION_COMMIT PROOF_PLAN_FILE_SHA256 h-iter-phase1-t0-model-free-proof-run1" >&2
  exit 64
fi
shift
[[ ${1:-} == --proof && $# -eq 4 ]]
readonly execution_commit=$2 plan_sha=$3 run_id=$4
readonly repo=/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1
readonly project=/home/ubuntu/rlm/prime-rl
readonly python_bin=/home/ubuntu/rlm/prime-rl/.venv/bin/python3
readonly plan_rel=experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1/t0-model-free-proof-plan.json
readonly sidecar_rel=experiments/qwen35-2b-latent-workspace-v1/h-iter-phase1-t0-train-calibration-v1/t0-model-free-proof-plan.sha256
readonly output=/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase1-t0-model-free-proof-run1
[[ $execution_commit =~ ^[0-9a-f]{40}$ && $plan_sha =~ ^[0-9a-f]{64}$ && $run_id == h-iter-phase1-t0-model-free-proof-run1 ]]
[[ -x $python_bin && -d $repo && -d $project/.venv && ! -e $output && ! -L $output ]]
[[ $(git -C "$repo" rev-parse HEAD) == "$execution_commit" ]]
[[ -z $(git -C "$repo" status --porcelain --untracked-files=all) ]]
[[ $(sha256sum "$repo/$plan_rel" | awk '{print $1}') == "$plan_sha" ]]
[[ $(tr -d '\n' < "$repo/$sidecar_rel") == "$plan_sha" ]]
[[ $(sha256sum "$project/pyproject.toml" | awk '{print $1}') == 504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656 ]]
[[ $(sha256sum "$project/uv.lock" | awk '{print $1}') == fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5 ]]
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src:$repo/scripts/latent" UV_PROJECT_ENVIRONMENT="$project/.venv"
set +e
"$python_bin" "$repo/scripts/latent/run_h_iter_phase1_t0_model_free_proof_v1.py" --repo "$repo" --execution-commit "$execution_commit" --plan-file-sha256 "$plan_sha" --run-id "$run_id" --output-dir "$output"
readonly proof_exit=$?
set -e
if [[ $proof_exit -eq 0 ]]; then
  "$python_bin" "$repo/scripts/latent/run_h_iter_phase1_t0_model_free_proof_v1.py" --repo "$repo" --execution-commit "$execution_commit" --plan-file-sha256 "$plan_sha" --run-id "$run_id" --output-dir "$output" --validate-terminal
  [[ -f $output/T0-MODEL-FREE-PROOF.json && ! -e $output/T0-MODEL-FREE-FAILURE.json ]]
  exit 0
fi
[[ -f $output/T0-MODEL-FREE-FAILURE.json && ! -e $output/T0-MODEL-FREE-PROOF.json ]]
"$python_bin" "$repo/scripts/latent/run_h_iter_phase1_t0_model_free_proof_v1.py" --repo "$repo" --execution-commit "$execution_commit" --plan-file-sha256 "$plan_sha" --run-id "$run_id" --output-dir "$output" --validate-terminal
exit "$proof_exit"
