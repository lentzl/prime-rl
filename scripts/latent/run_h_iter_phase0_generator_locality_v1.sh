#!/usr/bin/env bash
set -euo pipefail

if [[ ${1-} != --inner ]]; then
  if [[ $# -ne 3 ]]; then
    echo "usage: $0 <exact-execution-commit> <external-plan-sha256> <fresh-run-id>" >&2
    exit 64
  fi
  exec timeout --signal=TERM --kill-after=60s 1260s "$0" --inner "$@"
fi
shift

if [[ $# -ne 3 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^[0-9a-f]{64}$ || $3 != h-iter-phase0-generator-locality-run1 ]]; then
  echo "usage: $0 <exact-execution-commit> <external-plan-sha256> <fresh-run-id>" >&2
  exit 64
fi
readonly execution_commit=$1
readonly plan_sha256=$2
readonly run_id=$3
readonly repo=/home/ubuntu/rlm/worktrees/q35-2b-latent-workspace-v1
readonly plan="$repo/experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/phase0-plan.json"
readonly sidecar="$repo/experiments/qwen35-2b-latent-workspace-v1/h-iter-phase0-generator-locality-v1/phase0-plan.sha256"
readonly output_parent=/home/ubuntu/rlm/outputs
readonly output_dir=/home/ubuntu/rlm/outputs/q35-2b-h-iter-phase0-generator-locality-run1
readonly uv_bin=/home/ubuntu/.local/bin/uv
readonly shared_project=/home/ubuntu/rlm/prime-rl
readonly shared_venv="$shared_project/.venv"
readonly shared_pyproject_sha256=504907808f992f1e6883f54c2695a4814ae77d6b80814239cbfc98d81a543656
readonly shared_uv_lock_sha256=fca5fa6183345b5b68974078c38d58e0320f79eef13a695af11ceab12fdf36d5

for directory in "$repo" "$output_parent" "$shared_project" "$shared_venv"; do
  [[ -d "$directory" && ! -L "$directory" ]] || exit 2
done
[[ -x "$uv_bin" && -f "$plan" && ! -L "$plan" && -f "$sidecar" && ! -L "$sidecar" ]] || exit 2
[[ "$output_dir" == /home/ubuntu/rlm/outputs/q35-2b-h-iter-phase0-generator-locality-run1 ]] || exit 2
[[ -f "$shared_project/pyproject.toml" && ! -L "$shared_project/pyproject.toml" ]] || exit 2
[[ -f "$shared_project/uv.lock" && ! -L "$shared_project/uv.lock" ]] || exit 2
[[ "$(sha256sum "$shared_project/pyproject.toml" | cut -d' ' -f1)" == "$shared_pyproject_sha256" ]] || exit 2
[[ "$(sha256sum "$shared_project/uv.lock" | cut -d' ' -f1)" == "$shared_uv_lock_sha256" ]] || exit 2
[[ ! -e "$output_dir" && ! -L "$output_dir" ]] || exit 2
available_kib=$(df -Pk "$output_parent" | awk 'NR==2 {print $4}')
host_memory_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
(( available_kib >= 8388608 && host_memory_kib >= 8388608 )) || exit 2

cd "$repo"
[[ "$(git rev-parse HEAD)" == "$execution_commit" ]] || exit 2
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || exit 2
git cat-file -e a8f347c9a5fdf1c2d532c6527ce169cff0000a07^{commit}
git cat-file -e 4ae0308094a71d13520554da40cfe6375438b610^{commit}
[[ "$(sha256sum "$plan" | cut -d' ' -f1)" == "$plan_sha256" ]] || exit 2
[[ "$(cat "$sidecar")" == "$plan_sha256" ]] || exit 2

export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export UV_PROJECT_ENVIRONMENT="$shared_venv"
export PYTHONPATH="$repo/src"
set +e
"$uv_bin" run --project "$shared_project" --no-sync python \
  "$repo/scripts/latent/run_h_iter_phase0_generator_locality_v1.py" \
  --repo "$repo" --plan "$plan" --plan-file-sha256 "$plan_sha256" \
  --execution-commit "$execution_commit" --output-dir "$output_dir"
status=$?
set -e
[[ -d "$output_dir" && ! -L "$output_dir" ]] || exit 2
(( $(du -sb "$output_dir" | awk '{print $1}') <= 33554432 )) || exit 2
mapfile -t entries < <(find "$output_dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
if (( status == 0 )); then
  [[ "${entries[*]}" == "PROOF.json" ]] || exit 2
else
  [[ "${entries[*]}" == "FAILURE.json" ]] || exit 2
fi
exit "$status"
