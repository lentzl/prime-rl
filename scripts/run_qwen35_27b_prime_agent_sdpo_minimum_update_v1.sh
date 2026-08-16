#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${SDPO_MINIMUM_UPDATE_CONFIG:-$root/experiments/qwen35-27b-prime-agent-sdpo-v1/zero-lr-audit.toml}
model_revision=${MODEL_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}
audit_report=${SDPO_AUDIT_REPORT:-/ephemeral/outputs/qwen35-27b-prime-agent-sdpo-v1/zero-lr-audit/AUDIT.json}
run_dir=/ephemeral/outputs/qwen35-27b-prime-agent-sdpo-v1/minimum-update

cd "$root"
export PATH="$root/.venv/bin:$PATH"
# These PCIe-only L40S hosts lack CUDA peer access, while their SHM transport
# passes the full-size FSDP reduce-scatter probe. Avoid the socket fallback.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
if [[ ! -x .venv/bin/rl || ! -x .venv/bin/vllm-router ]]; then
  echo "Prime-RL training executables are missing" >&2
  exit 1
fi
if [[ ! -f "$config" || ! -f "$audit_report" ]]; then
  echo "minimum update requires its base config and passing audit report" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "refusing to launch while another GPU process is active" >&2
  exit 1
fi
.venv/bin/python - "$config" "$audit_report" "$model_revision" <<'PY'
import json
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
report = json.load(open(sys.argv[2]))
if config.get("max_steps") != 1 or config.get("trainer", {}).get("optim", {}).get("lr") != 0.0:
    raise SystemExit("minimum update must derive from the one-step zero-LR config")
if "ckpt" in config:
    raise SystemExit("minimum update base config must leave checkpointing disabled")
if (
    report.get("verdict") != "pass"
    or report.get("mechanism") != "mixed-feedback-conditioned-sdpo-grpo-zero-lr"
    or report.get("expected_revision") != sys.argv[3]
):
    raise SystemExit("minimum update requires a matching passing zero-LR audit")
PY

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export HF_TOKEN=${HF_TOKEN:-${HF_KEY:-}}
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN or HF_KEY is required" >&2
  exit 1
fi
model_snapshot=$(
  .venv/bin/python - "$model_revision" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download("Qwen/Qwen3.5-27B", revision=sys.argv[1]))
PY
)
if [[ "$(basename "$model_snapshot")" != "$model_revision" ]]; then
  echo "resolved model snapshot does not match the pinned revision" >&2
  exit 1
fi

args=(
  rl @ "$config"
  --model.name "$model_snapshot"
  --run.name minimum-update
  --run.dir minimum-update
  --trainer.optim.lr 1e-7
  --trainer.ckpt.weights-only true
  --trainer.ckpt.interval 1
  --trainer.ckpt.keep-last 1
  --orchestrator.ckpt.interval 1
  --orchestrator.ckpt.keep-last 1
)
if [[ "${SDPO_MINIMUM_UPDATE_DRY_RUN:-false}" == true ]]; then
  "${args[@]}" --dry-run
  echo "minimum SDPO update preflight passed"
  exit 0
fi

"${args[@]}"

.venv/bin/python scripts/finalize_hf_processor_metadata.py \
  "$model_snapshot" \
  "$run_dir/weights/step_1"
.venv/bin/python scripts/validate_prime_agent_sdpo_minimum_update_v1.py \
  "$run_dir" \
  --expected-revision "$model_revision" \
  --audit-report "$audit_report" \
  --output "$run_dir/UPDATE.json" | tee "$run_dir/UPDATE.txt"
