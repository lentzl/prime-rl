#!/usr/bin/env bash
set -euo pipefail

readonly WORKTREE=/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1
readonly UV=/home/ubuntu/.local/bin/uv
readonly UV_PROJECT=/home/ubuntu/rlm/prime-rl
readonly SHARED_ENV=/home/ubuntu/rlm/prime-rl/.venv
readonly EXPERIMENT_DIR="$WORKTREE/experiments/qwen35-2b-latent-coordinator-v1"
readonly OUTPUT_DIR=/home/ubuntu/rlm/results/q35-2b-phase-b-fixed-depth-smoke-a0c-br5

if [[ $# -lt 2 || $# -gt 3 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: $0 <exact-clean-execution-commit> <root-authorized-repair-plan-sha256> [--preflight-only]" >&2
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
[[ -z $(git status --porcelain --untracked-files=all) ]]
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]]
[[ $(sha256sum "$EXPERIMENT_DIR/phase-b-fixed-depth-smoke-a0c-br5-plan.json" | cut -d' ' -f1) == "$ROOT_AUTHORIZED_PLAN_SHA256" ]]
(cd "$EXPERIMENT_DIR" && sha256sum --check phase-b-fixed-depth-smoke-a0c-br5.sha256)

export UV_PROJECT_ENVIRONMENT="$SHARED_ENV"
export PYTHONPATH="$WORKTREE/src:$WORKTREE/packages/prime-rl-configs/src"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The Python compute bound is 114 minutes, followed by at most five minutes
# of preservation hashing and one reserved minute for terminal publication.
# The independent outer timeout is unchanged from the rejected prior start.
exec timeout --signal=TERM --kill-after=30s 120m \
  "$UV" run --project "$UV_PROJECT" --no-sync python \
  scripts/latent/run_phase_b_fixed_depth_smoke_v1.py \
  --plan "$EXPERIMENT_DIR/phase-b-fixed-depth-smoke-a0c-br5-plan.json" \
  --selection "$EXPERIMENT_DIR/phase-b-fixed-depth-smoke-v1-selection.json" \
  --a0c-binding "$EXPERIMENT_DIR/phase-b-a0c-binding-v1.json" \
  --a0c-binding-hash "$EXPERIMENT_DIR/phase-b-a0c-binding-v1.sha256" \
  --output-dir "$OUTPUT_DIR" \
  --execution-commit "$EXECUTION_COMMIT" \
  ${MODE:+"$MODE"}
