#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! $1 =~ ^[0-9a-f]{40}$ || $2 != a1-nc0-cap768-r3-render-proof-run1 ]]; then
  echo "usage: $0 <exact-execution-commit> <exact-proof-run-id>" >&2
  exit 64
fi
readonly execution_commit=$1 run_id=$2
readonly repo=/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1
readonly experiment="$repo/experiments/qwen35-2b-latent-workspace-v1"
readonly coordinator=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
readonly output_root=/home/ubuntu/rlm/outputs/latent-a1-nc0-cap768-r3-render-proof-v1
readonly output_dir="$output_root/$run_id"
readonly uv_bin=/home/ubuntu/.local/bin/uv
readonly shared_project=/home/ubuntu/rlm/prime-rl
readonly shared_venv="$shared_project/.venv"

[[ -d "$repo" && ! -L "$repo" && -d "$coordinator" && ! -L "$coordinator" ]] || exit 2
[[ -x "$uv_bin" && -d "$shared_venv" && ! -e "$output_dir" && ! -L "$output_root" ]] || exit 2
cd "$repo"
[[ "$(git rev-parse HEAD)" == "$execution_commit" && -z "$(git status --porcelain --untracked-files=all)" ]] || exit 2
install -d -m 700 "$output_root" "$output_dir"
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$shared_venv" PYTHONPATH="$repo/src"
set +e
timeout --signal=TERM --kill-after=30s 600s "$uv_bin" run --project "$shared_project" --no-sync python \
  "$repo/scripts/latent/prove_a1_nc0_cap768_r3_render_v1.py" --repo "$repo" \
  --coordinator "$coordinator" --train-bank "$experiment/a1-nc0-train-bank-v1.json" \
  --held-out-bank "$experiment/a1-nc0-held_out-bank-v1.json" \
  --output-dir "$output_dir" --execution-commit "$execution_commit" \
  >"$output_dir/proof.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$output_dir/exit_status.txt"
if (( status == 0 )); then
  [[ -f "$output_dir/receipt.json" && ! -L "$output_dir/receipt.json" ]] || exit 2
fi
exit "$status"
