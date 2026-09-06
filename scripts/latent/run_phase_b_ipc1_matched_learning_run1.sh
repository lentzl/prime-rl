#!/usr/bin/env bash
set -euo pipefail

readonly WORKTREE=/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1
readonly UV=/home/ubuntu/.local/bin/uv
readonly UV_PROJECT=/home/ubuntu/rlm/prime-rl
readonly SHARED_ENV=/home/ubuntu/rlm/prime-rl/.venv
readonly EXPERIMENT_DIR="$WORKTREE/experiments/qwen35-2b-latent-coordinator-v1"
readonly PLAN="$EXPERIMENT_DIR/phase-b-ipc1-matched-learning-run1-plan.json"
readonly FREEZE_MANIFEST="$EXPERIMENT_DIR/phase-b-ipc1-matched-learning-run1.sha256"
readonly OUTPUT_DIR=/home/ubuntu/rlm/results/q35-2b-b-ipc1-matched-learning-run1

if [[ $# -lt 2 || $# -gt 3 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: $0 <exact-clean-execution-commit> <root-authorized-plan-sha256> [--preflight-only]" >&2
  exit 64
fi
readonly EXECUTION_COMMIT=$1
readonly ROOT_AUTHORIZED_PLAN_SHA256=$2
readonly MODE=${3:-}
if [[ -n $MODE && $MODE != --preflight-only ]]; then
  echo "third argument must be --preflight-only" >&2
  exit 64
fi

cd "$WORKTREE"
[[ $(git rev-parse HEAD) == "$EXECUTION_COMMIT" ]]
[[ $(git rev-parse "$EXECUTION_COMMIT^") == 69517c54c283017d1e8b76e1afda7e36adf2552f ]]
[[ -z $(git status --porcelain --untracked-files=all) ]]
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]') ]]
[[ $(sha256sum "$PLAN" | cut -d' ' -f1) == "$ROOT_AUTHORIZED_PLAN_SHA256" ]]
(cd "$EXPERIMENT_DIR" && sha256sum --check "$(basename "$FREEZE_MANIFEST")")

export UV_PROJECT_ENVIRONMENT="$SHARED_ENV"
export PYTHONPATH="$WORKTREE/src:$WORKTREE/packages/prime-rl-configs/src"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec timeout --signal=TERM --kill-after=60s 4h \
  "$UV" run --project "$UV_PROJECT" --no-sync python \
  scripts/latent/run_phase_b_ipc1_matched_learning_v1.py \
  --plan "$PLAN" \
  --output-dir "$OUTPUT_DIR" \
  --execution-commit "$EXECUTION_COMMIT" \
  --authorized-plan-sha256 "$ROOT_AUTHORIZED_PLAN_SHA256" \
  ${MODE:+"$MODE"}
