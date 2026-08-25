#!/usr/bin/env bash
set -euo pipefail

root=/home/ubuntu/rlm/prime-rl
export UV_BIN=/home/ubuntu/.local/bin/uv
export PATH=/home/ubuntu/.local/bin:$root/.venv/bin:$PATH
export QWEN38_QUALIFICATION_OUTPUT_ROOT=/home/ubuntu/rlm/results/q35-2b-self-bootstrap-dual-dense-v1
export QWEN38_QUALIFICATION_AXES=natural_n1a
export QWEN38_QUALIFICATION_NUM_TASKS=6
export QWEN38_QUALIFICATION_NUM_ROLLOUTS=1
export QWEN38_QUALIFICATION_MAX_CONCURRENT=6
export QWEN38_QUALIFICATION_START_INDEX=4500000
export QWEN38_QUALIFICATION_EVAL_MAX_ADDRESS_SPACE_BYTES=34359738368
export QUALIFICATION_REASONING_EFFORT=high
export QUALIFICATION_SAMPLING_SEED=20260823
export QUALIFICATION_SAMPLING_TEMPERATURE=0.6
export QUALIFICATION_PRIVILEGED_BOOTSTRAP_PATH=/home/ubuntu/rlm/artifacts/q35-2b-self-bootstrap-dual-dense-v1/mixed-c5-k4-yield-e0d3x-4500000-n6-bootstrap.json
export PROCEDURAL_INTERACTION_CURRICULUM=e0d3_uncapped_yield_exact_child
export DUAL_EXTERNAL_MODEL=q35-dual-mixed-c5-k4

cd "$root"
exec scripts/run_q35_2b_dual_policy_mastery_v1.sh \
  /home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-v1/dual-dense-coordinator-c4-to-c5-0019ab505e6c/weights/step_1 \
  /home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-v1/dual-dense-child-k3-to-k4-908b06660ab3/weights/step_1 \
  mixed-c5-k4-yield-e0d3x-4500000-n6-attempt2 \
  local
