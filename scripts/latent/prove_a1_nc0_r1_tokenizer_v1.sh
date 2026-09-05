#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^a1-nc0-r1-tokenizer-proof-[a-z0-9][a-z0-9._-]{2,48}$ ]]; then
  echo "usage: $0 <exact-clean-mechanism-commit> <fresh-proof-run-id>" >&2
  exit 64
fi
readonly mechanism_commit=$1
readonly run_id=$2
readonly repo=/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1
readonly experiment="$repo/experiments/qwen35-2b-latent-workspace-v1"
readonly coordinator=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
readonly output_root=/home/ubuntu/rlm/outputs/latent-a1-nc0-r1-tokenizer-proof-v1
readonly output_dir="$output_root/$run_id"
readonly uv_bin=/home/ubuntu/.local/bin/uv
readonly shared_project=/home/ubuntu/rlm/prime-rl
readonly shared_venv="$shared_project/.venv"

[[ -d "$repo" && ! -L "$repo" && -d "$coordinator" && ! -L "$coordinator" ]] || exit 2
[[ -x "$uv_bin" && -d "$shared_venv" && ! -L "$shared_venv" ]] || exit 2
[[ ! -L "$output_root" && ! -e "$output_dir" && ! -L "$output_dir" ]] || exit 2
cd "$repo"
[[ "$(git rev-parse HEAD)" == "$mechanism_commit" ]] || exit 2
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || exit 2
install -d -m 700 "$output_root"
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$shared_venv" PYTHONPATH="$repo/src"
"$uv_bin" run --project "$shared_project" --no-sync python \
  "$repo/scripts/latent/prove_a1_nc0_r1_tokenizer_v1.py" \
  --repo "$repo" --tokenizer "$coordinator" \
  --train-bank "$experiment/a1-nc0-train-bank-v1.json" \
  --validation-bank "$experiment/a1-nc0-validation-bank-v1.json" \
  --held-out-bank "$experiment/a1-nc0-held_out-bank-v1.json" \
  --output-dir "$output_dir" --mechanism-commit "$mechanism_commit"
mapfile -t entries < <(find "$output_dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
[[ "${entries[*]}" == "proof.log receipt.json" ]] || exit 2
