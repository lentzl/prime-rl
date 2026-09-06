#!/usr/bin/env bash
set -euo pipefail

readonly WORKTREE=/home/ubuntu/rlm/worktrees/q35-2b-recurrent-sidecar-v1
readonly UV=/home/ubuntu/.local/bin/uv
readonly UV_PROJECT=/home/ubuntu/rlm/prime-rl
readonly SHARED_ENV=/home/ubuntu/rlm/prime-rl/.venv
readonly EXPERIMENT_DIR="$WORKTREE/experiments/qwen35-2b-latent-coordinator-v1"
readonly PLAN="$EXPERIMENT_DIR/phase-b-ipc1-render-proof-hygiene-v1-plan.json"
readonly FREEZE_MANIFEST="$EXPERIMENT_DIR/phase-b-ipc1-render-proof-hygiene-v1.sha256"
readonly OUTPUT_DIR=/home/ubuntu/rlm/outputs/q35-2b-ipc1-render-proof-hygiene-v1-run1
readonly MECHANISM_COMMIT=bb695195b950946c727b4bb17d46580f55d1bf15

if [[ $# -ne 3 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^[0-9a-f]{64}$ || ! $3 =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: $0 <exact-clean-proof-commit> <root-authorized-plan-sha256> <root-authorized-freeze-sha256>" >&2
  exit 64
fi
readonly EXECUTION_COMMIT=$1
readonly ROOT_AUTHORIZED_PLAN_SHA256=$2
readonly ROOT_AUTHORIZED_FREEZE_SHA256=$3

cd "$WORKTREE"
[[ $(git rev-parse HEAD) == "$EXECUTION_COMMIT" ]]
[[ $(git rev-parse "$EXECUTION_COMMIT^") == "$MECHANISM_COMMIT" ]]
[[ -z $(git status --porcelain --untracked-files=all) ]]
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]]
[[ $(sha256sum "$PLAN" | cut -d' ' -f1) == "$ROOT_AUTHORIZED_PLAN_SHA256" ]]
[[ $(sha256sum "$FREEZE_MANIFEST" | cut -d' ' -f1) == "$ROOT_AUTHORIZED_FREEZE_SHA256" ]]
(cd "$EXPERIMENT_DIR" && sha256sum --check "$(basename "$FREEZE_MANIFEST")")

export UV_PROJECT_ENVIRONMENT="$SHARED_ENV"
export PYTHONPATH="$WORKTREE/src:$WORKTREE/packages/prime-rl-configs/src"
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec timeout --signal=TERM --kill-after=30s 10m \
  "$UV" run --project "$UV_PROJECT" --no-sync python \
  "$WORKTREE/scripts/latent/prove_phase_b_ipc1_render_hygiene_v1.py" \
  --plan "$PLAN" \
  --freeze-manifest "$FREEZE_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --execution-commit "$EXECUTION_COMMIT" \
  --authorized-plan-sha256 "$ROOT_AUTHORIZED_PLAN_SHA256" \
  --authorized-freeze-sha256 "$ROOT_AUTHORIZED_FREEZE_SHA256"
