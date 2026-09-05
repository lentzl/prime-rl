#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revision=${1:?exact clean execution revision required}
verifiers_revision=${2:?exact clean durable Verifiers revision required}
config_sha=${3:?root-frozen P0 config SHA-256 required}
dry_run=${P0_DRY_RUN:-false}

config=experiments/qwen35-2b-document-recursion-zero-update-v1/\
specialist-source-route-parity-p0-lr0.toml
validator=scripts/validate_q35_2b_source_route_parity_p0_v1.py

e33=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/\
c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
h176=/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/\
h176child8-document-child-real12-step8-v2/weights/step_8
s2=/home/ubuntu/rlm/outputs/q35-2b-specialist-competence-s2-v1/\
h-source-s2-step8-v1/weights/step_8
s5=/home/ubuntu/rlm/outputs/q35-2b-source-worker-remedial-s5-v1/\
h-source-s5-remedial-step8-v1/weights/step_8

e33_sha=e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47
h176_sha=77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e
s2_sha=937a96154cd47d8dda1fb01125d9f552037a91bc0542748f417652ceddd47f47
s5_sha=09a2e3e88030d17896e554211d8fc7eff709d6b4a619e99e3342d05aacde0782

output_root=/home/ubuntu/rlm/outputs/q35-2b-source-route-parity-p0-v1
run_dir=$output_root/lr0-calibration/source-route-parity-p0-lr0-calibration
result_root=/home/ubuntu/rlm/results/q35-2b-source-route-parity-p0-v1
route_audit=$result_root/routing-audit.jsonl
calibration_result=$result_root/calibration.json
legacy_verifiers_gitlink=5283a85a01b5e8a065b3d2db17f9efa6aa0f3b2f
live_detached_verifiers_prefix=53bafca
uv_bin=/home/ubuntu/.local/bin/uv
uv_environment=/home/ubuntu/rlm/prime-rl/.venv
p0_pythonpath="$root/src:$root/packages/prime-rl-configs/src:$root/deps/verifiers/environments/subagent_communication_v1:$root/deps/verifiers/environments/source_worker_first_call_v1"
export PYTHONPATH="$p0_pythonpath${PYTHONPATH:+:$PYTHONPATH}"

cd "$root"
if [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
  echo "P0 execution revision mismatch" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "P0 requires a clean parent worktree" >&2
  exit 1
fi
if [[ "$(git -C deps/verifiers rev-parse --show-toplevel 2>/dev/null)" != "$root/deps/verifiers" ]]; then
  echo "P0 requires an initialized Verifiers submodule" >&2
  exit 1
fi
actual_verifiers_revision=$(git -C deps/verifiers rev-parse HEAD)
if [[ "$actual_verifiers_revision" != "$verifiers_revision" ]]; then
  echo "P0 Verifiers revision mismatch" >&2
  exit 1
fi
if [[ "$actual_verifiers_revision" == "$legacy_verifiers_gitlink" || "$actual_verifiers_revision" == "$live_detached_verifiers_prefix"* ]]; then
  echo "P0 refuses legacy or detached-live Verifiers revisions" >&2
  exit 1
fi
if [[ -n "$(git -C deps/verifiers status --porcelain)" ]]; then
  echo "P0 requires a clean durable Verifiers checkout" >&2
  exit 1
fi
for package in subagent_communication_v1 source_worker_first_call_v1; do
  if [[ ! -f "$root/deps/verifiers/environments/$package/pyproject.toml" ]]; then
    echo "P0 Verifiers package is missing: $package" >&2
    exit 1
  fi
done
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ \
      || ! "$verifiers_revision" =~ ^[0-9a-f]{40}$ \
      || ! "$config_sha" =~ ^[0-9a-f]{64}$ ]]; then
  echo "P0 requires exact revision and config identities" >&2
  exit 1
fi
test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha"

rehash_protected_models() {
  for model_and_sha in \
    "$e33:$e33_sha" \
    "$h176:$h176_sha" \
    "$s2:$s2_sha" \
    "$s5:$s5_sha"; do
    model=${model_and_sha%:*}
    expected_sha=${model_and_sha##*:}
    test -f "$model/STABLE"
    test "$(sha256sum "$model/model.safetensors" | awk '{print $1}')" = "$expected_sha"
  done
}

rehash_protected_models
if [[ -e "$output_root" || -e "$result_root" ]]; then
  echo "refusing duplicate or partial P0 output/result root" >&2
  exit 1
fi

run_p0_python() {
  UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync python "$@"
}
run_p0_rl() {
  UV_PROJECT_ENVIRONMENT="$uv_environment" "$uv_bin" run --no-sync rl @ "$@"
}
run_p0_python "$validator" "$config"

if [[ "$dry_run" == true ]]; then
  run_p0_rl "$config" --dry-run
  echo "P0 static and launcher dry-run checks passed"
  exit 0
fi

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ ${#gpu_names[@]} -ne 2 ]] || printf '%s\n' "${gpu_names[@]}" | grep -qv 'RTX A6000'; then
  echo "P0 requires exactly 2x RTX A6000" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing P0 launch while another GPU process is active" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export PATH="$uv_environment/bin:$HOME/.local/bin:$PATH"
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

run_p0_rl "$config"
if [[ ! -f "$run_dir/metrics.jsonl" \
      || ! -f "$run_dir/rollouts/step_1/train/all/traces.jsonl" \
      || ! -f "$run_dir/rollouts/step_1/train/effective/traces.jsonl" \
      || ! -f "$route_audit" \
      || -e "$calibration_result" ]]; then
  echo "P0 did not produce one complete unrecorded calibration" >&2
  exit 1
fi
if find "$run_dir/checkpoints" "$run_dir/weights" -type f -print -quit 2>/dev/null | grep -q .; then
  echo "P0 wrote a forbidden model artifact" >&2
  exit 1
fi
rehash_protected_models
run_p0_python "$validator" "$run_dir" --runtime \
  --execution-revision "$revision" \
  --verifiers-revision "$verifiers_revision" \
  --config-sha256 "$config_sha" \
  --output "$calibration_result"
rehash_protected_models
test -f "$calibration_result"
echo "P0 calibration recorded; no checkpoint or optimizer authorization: $calibration_result"
