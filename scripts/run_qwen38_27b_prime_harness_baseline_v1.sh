#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model=${1:-Qwen/Qwen3.8-27B}
label=${2:-untouched-qwen38-27b}
revision=${3:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}

EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0,1} \
EVAL_TENSOR_PARALLEL_SIZE=${EVAL_TENSOR_PARALLEL_SIZE:-2} \
EVAL_DTYPE=${EVAL_DTYPE:-bfloat16} \
EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-32768} \
EVAL_GPU_MEMORY_UTILIZATION=${EVAL_GPU_MEMORY_UTILIZATION:-0.88} \
EVAL_MAX_NUM_SEQS=${EVAL_MAX_NUM_SEQS:-1} \
EVAL_MAX_NUM_BATCHED_TOKENS=${EVAL_MAX_NUM_BATCHED_TOKENS:-512} \
EVAL_LANGUAGE_MODEL_ONLY=${EVAL_LANGUAGE_MODEL_ONLY:-true} \
EVAL_DISABLE_CUSTOM_ALL_REDUCE=${EVAL_DISABLE_CUSTOM_ALL_REDUCE:-true} \
EVAL_DRIVER=${EVAL_DRIVER:-scripts/run_qwen38_27b_prime_harness_qualification_v1.sh} \
EVAL_EXPERIMENT_DIR=${EVAL_EXPERIMENT_DIR:-experiments/qwen38-27b-prime-harness-qualification-v1} \
PRIME_MASTERY_OUTPUT_ROOT=${PRIME_MASTERY_OUTPUT_ROOT:-/ephemeral/evals/qwen38-27b-prime-harness-qualification-v1} \
  "$root/scripts/run_qwen35_27b_prime_agent_mastery_baseline_v2.sh" \
  "$model" "$label" "$revision"
