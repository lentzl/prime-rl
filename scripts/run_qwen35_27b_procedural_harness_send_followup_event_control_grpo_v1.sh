#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
label=${1:-r1}

export HARNESS_CUMULATIVE_FOLLOWUP_REWARD_MODE=event_control
export HARNESS_CUMULATIVE_FOLLOWUP_SUPPORT_TRACES=${HARNESS_CUMULATIVE_FOLLOWUP_SUPPORT_TRACES:-/ephemeral/outputs/qwen35-27b-procedural-harness-action-ramp-v1/send-followup-cumulative-grpo-r2/rollouts/step_1/train/all/traces.jsonl}
export HARNESS_CUMULATIVE_SEND_START_INDEX=${HARNESS_CUMULATIVE_SEND_START_INDEX:-1500000}
export HARNESS_CUMULATIVE_FOLLOWUP_START_INDEX=${HARNESS_CUMULATIVE_FOLLOWUP_START_INDEX:-1600000}

exec "$root/scripts/run_qwen35_27b_procedural_harness_send_followup_cumulative_grpo_v1.sh" \
  "event-control-$label"
