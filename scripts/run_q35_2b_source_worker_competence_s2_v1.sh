#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
revision=${1:?exact execution revision required}

e33=/home/ubuntu/rlm/outputs/q35-2b-adaptive-cognition-sft-v1/c54-step8-action4-adaptive-nonroot-step2-v4/weights/step_2
h176=/home/ubuntu/rlm/outputs/q35-2b-document-child-sft-v1/h176child8-document-child-real12-step8-v2/weights/step_8
dataset=/home/ubuntu/rlm/artifacts/q35-2b-specialist-population-c1-v1/source-inspector-sft-v1
output_root=/home/ubuntu/rlm/outputs/q35-2b-specialist-competence-s2-v1
result_root=/home/ubuntu/rlm/results/q35-2b-specialist-competence-s2-v1
state_dir=$result_root/state
config=experiments/qwen35-2b-document-recursion-zero-update-v1/specialist-source-competence-s2-heldout-v1.toml
sampling_contract=experiments/qwen35-2b-document-recursion-zero-update-v1/specialist-competence-s1-sampling-contract.json
candidate=$output_root/h-source-s2-step8-v1/weights/step_8
control_label=s2-source-h176-forced-control-v1
treatment_label=s2-source-h-source-forced-treatment-v1
summary=$state_dir/s2-source-forced-paired-summary-v1.json

e33_sha=e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47
h176_sha=77980e247bbccd6463ddda02cd42d2c357e15f8ec1ad0ea84627e008a8674a1e
dataset_sha=01d7359cac90f5664ff0364520b3c98cc31227c56496b1d78f704712f9a5bab3
config_sha=97c37192a9729fbfb92782b78807a5fa150499645c152eaf9c0a24855911d431
sampling_sha=10fd257ca94450086fb750301cbdb3c3b30d51f35aac4e761e19cb49d921999c
task_bank_sha=9c38f1dff3787222908e896a151f0889edeeefea7b781f8450736f4b8cc06bd2

cd "$root"
if [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
  echo "execution revision mismatch" >&2
  exit 1
fi
git merge-base --is-ancestor 5283a85a01b5e8a065b3d2db17f9efa6aa0f3b2f HEAD
test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha"
test "$(sha256sum "$sampling_contract" | awk '{print $1}')" = "$sampling_sha"
test "$(sha256sum "$e33/model.safetensors" | awk '{print $1}')" = "$e33_sha"
test "$(sha256sum "$h176/model.safetensors" | awk '{print $1}')" = "$h176_sha"
test "$(sha256sum "$dataset/train.parquet" | awk '{print $1}')" = "$dataset_sha"

for path in \
  "$result_root/$control_label" \
  "$result_root/$treatment_label" \
  "$output_root/h-source-s2-step8-v1" \
  "$state_dir/h-source-s2-step8-v1-receipt.json" \
  "$summary"; do
  if [[ -e "$path" ]]; then
    echo "refusing duplicate S2 artifact: $path" >&2
    exit 1
  fi
done

UV_PROJECT_ENVIRONMENT=/home/ubuntu/rlm/prime-rl/.venv \
  /home/ubuntu/.local/bin/uv run --no-sync python \
  scripts/hash_q35_2b_specialist_task_bank_v1.py "$config" \
  --expected-sha256 "$task_bank_sha"

common_env=(
  PRIME_API_KEY=prime
  UV_PROJECT_ENVIRONMENT=/home/ubuntu/rlm/prime-rl/.venv
  UV_BIN=/home/ubuntu/.local/bin/uv
  INFERENCE_BIN=/home/ubuntu/rlm/prime-rl/.venv/bin/inference
  EVAL_BIN=/home/ubuntu/rlm/prime-rl/.venv/bin/eval
  EVAL_PYTHON_BIN=/home/ubuntu/rlm/prime-rl/.venv/bin/python
  SPECIALIST_COMPETENCE_OUTPUT_ROOT="$result_root"
  SPECIALIST_COMPETENCE_FORCE_FIXED_ACTION=1
)

env "${common_env[@]}" scripts/run_q35_2b_specialist_competence_eval_v1.sh \
  "$e33" "$h176" "$h176" source_inspector "$control_label" "$revision" "$config"

UV_PROJECT_ENVIRONMENT=/home/ubuntu/rlm/prime-rl/.venv \
  /home/ubuntu/.local/bin/uv run --no-sync python \
  scripts/run_q35_2b_specialist_worker_sft_v1.py \
  --repo "$root" \
  --source-model "$h176" \
  --dataset-dir "$dataset" \
  --output-root "$output_root" \
  --state-dir "$state_dir" \
  --run-name h-source-s2-step8-v1 \
  --expert-id source_inspector \
  --learning-rate 2e-6 \
  --optimizer-updates 8 \
  --batch-size 16

env "${common_env[@]}" scripts/run_q35_2b_specialist_competence_eval_v1.sh \
  "$e33" "$h176" "$candidate" source_inspector "$treatment_label" "$revision" "$config"

UV_PROJECT_ENVIRONMENT=/home/ubuntu/rlm/prime-rl/.venv \
  /home/ubuntu/.local/bin/uv run --no-sync python \
  scripts/summarize_q35_2b_specialist_competence_v1.py \
  --expert-id source_inspector \
  --control-traces "$result_root/$control_label/document/document/traces.jsonl" \
  --control-audit "$result_root/$control_label/ROUTING_AUDIT.jsonl" \
  --treatment-traces "$result_root/$treatment_label/document/document/traces.jsonl" \
  --treatment-audit "$result_root/$treatment_label/ROUTING_AUDIT.jsonl" \
  --control-model "$h176" \
  --treatment-model "$candidate" \
  --expected-tasks 16 \
  --minimum-worker-activations 16 \
  --minimum-treatment-hard-successes 4 \
  --minimum-hard-successes-per-family 2 \
  --minimum-paired-recoveries 4 \
  --maximum-paired-regressions 0 \
  --require-forced-assignment \
  --output "$summary"

test "$(sha256sum "$e33/model.safetensors" | awk '{print $1}')" = "$e33_sha"
test "$(sha256sum "$h176/model.safetensors" | awk '{print $1}')" = "$h176_sha"
echo "source worker competence S2 completed: $summary"
