#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
role=${1:-}
student_snapshot=${2:-}
adapter_path=${3:-}
label=${4:-}
revision=${5:-local}
expected_student_sha=c75915dd41cd4fc9b1a1ef5582c6fd14913fc6f9971a58feca3b72b4bfcad406

case "$role" in
  orchestrator|child) ;;
  *) echo "usage: $0 {orchestrator|child} STUDENT_SNAPSHOT ADAPTER_PATH LABEL [REVISION]" >&2; exit 1 ;;
esac
if [[ -z "$student_snapshot" || -z "$adapter_path" || -z "$label" ]]; then
  echo "usage: $0 {orchestrator|child} STUDENT_SNAPSHOT ADAPTER_PATH LABEL [REVISION]" >&2
  exit 1
fi
if [[ ! -f "$student_snapshot/STABLE" || ! -f "$student_snapshot/model.safetensors" ]]; then
  echo "student snapshot is not a stable dense export: $student_snapshot" >&2
  exit 1
fi
actual_student_sha=$(sha256sum "$student_snapshot/model.safetensors" | awk '{print $1}')
if [[ "$actual_student_sha" != "$expected_student_sha" ]]; then
  echo "student weight hash mismatch: $actual_student_sha" >&2
  exit 1
fi
if [[ ! -f "$adapter_path/adapter_config.json" \
  || ! -f "$adapter_path/adapter_model.safetensors" \
  || ! -f "$adapter_path/../STABLE" ]]; then
  echo "adapter is not a complete stable Prime-RL checkpoint: $adapter_path" >&2
  exit 1
fi

cd "$root"
export PATH="$HOME/.local/bin:$root/.venv/bin:$PATH"
uv run --frozen --no-sync python - "$student_snapshot" "$adapter_path" <<'PY'
import json
import sys
from pathlib import Path

student = Path(sys.argv[1]).resolve()
adapter = Path(sys.argv[2]).resolve()
config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
    raise SystemExit("adapter is not a causal-LM LoRA")
if config.get("r") != 16:
    raise SystemExit(f"adapter rank is {config.get('r')}; expected 16")
base = config.get("base_model_name_or_path")
if not isinstance(base, str) or Path(base).resolve() != student:
    raise SystemExit("adapter config identifies a different base model")
expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
if set(config.get("target_modules") or ()) != expected_targets:
    raise SystemExit("adapter target-module set does not match the preregistered topology")
if config.get("modules_to_save") not in (None, []):
    raise SystemExit("adapter contains non-LoRA trainable modules")
PY

adapter_sha=$(sha256sum "$adapter_path/adapter_model.safetensors" | awk '{print $1}')
lora_name="q35-2b-${role}-r16-${adapter_sha:0:12}"
EVAL_SERVED_MODEL=$lora_name \
EVAL_LORA_NAME=$lora_name \
EVAL_LORA_PATH=$(cd "$adapter_path" && pwd) \
EVAL_MAX_LORA_RANK=16 \
  "$root/scripts/run_q35_2b_role_distillation_eval_v1.sh" \
  "$student_snapshot" "$label" "$revision"
