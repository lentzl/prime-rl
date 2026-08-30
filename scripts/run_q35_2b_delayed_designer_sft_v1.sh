#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
role=${1:?role required}
source_model=${2:?source model required}
generation=${3:?generation artifact required}
score=${4:?score artifact required}
corpus_dir=${5:?designer corpus directory required}
run_name=${6:?run name required}
output_root=${Q35_2B_ROLE_GRPO_OUTPUT_ROOT:-/home/ubuntu/rlm/outputs/q35-2b-self-bootstrap-dual-dense-grpo-v1}
experiment_dir=${Q35_2B_ROLE_GRPO_EXPERIMENT_DIR:-$root/experiments/qwen35-2b-self-bootstrap-dual-dense-v1/grpo-runs}
uv_bin=${UV_BIN:-/home/ubuntu/.local/bin/uv}
learning_rate=${Q35_2B_DESIGNER_SFT_LR:-1e-6}
revision=${MODEL_REVISION:-main}

case "$role" in coordinator|child) ;; *) echo "invalid Designer role: $role" >&2; exit 1 ;; esac
for path in "$generation" "$score"; do
  [[ "$path" == /* && -f "$path" ]] || { echo "missing absolute Designer artifact: $path" >&2; exit 1; }
done
[[ "$source_model" == /* && -f "$source_model/STABLE" && -f "$source_model/model.safetensors" ]] || {
  echo "incomplete Designer source checkpoint: $source_model" >&2
  exit 1
}
[[ "$run_name" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || { echo "invalid run name: $run_name" >&2; exit 1; }

cd "$root"
export PATH="$root/.venv/bin:$HOME/.local/bin:$PATH"
mkdir -p "$experiment_dir" "$(dirname "$corpus_dir")"
config=$experiment_dir/$run_name.toml
receipt=$experiment_dir/$run_name-receipt.json
run_output=$output_root/$run_name
for target in "$corpus_dir" "$config" "$receipt" "$run_output"; do
  [[ ! -e "$target" ]] || { echo "refusing duplicate delayed Designer target: $target" >&2; exit 1; }
done

source_sha=$(sha256sum "$source_model/model.safetensors" | awk '{print $1}')
"$uv_bin" run --frozen --no-sync scripts/q35_2b_spade_coevolution_v1.py export-designer \
  --generation "$generation" \
  --score "$score" \
  --output-dir "$corpus_dir" \
  --student-snapshot "$source_model" \
  --student-revision "$revision" \
  --student-weight-sha "$source_sha" \
  --max-rows 2

"$uv_bin" run --frozen --no-sync python - \
  "$corpus_dir/MANIFEST.json" "$role" "$source_sha" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if (
    manifest.get("role") != sys.argv[2]
    or manifest.get("objective") != "environment_designer"
    or manifest.get("training_stage") != "delayed_reward_filtered_coevolution"
    or manifest.get("student", {}).get("weight_sha256") != sys.argv[3]
    or not 1 <= manifest.get("rows", 0) <= 2
    or manifest.get("exact_answer_rows") != 0
):
    raise SystemExit("invalid delayed role-local Designer corpus")
PY

cat >"$config" <<EOF
max_steps = 1
output_dir = "$output_root"
clean = false

[run]
name = "$run_name"
dir = "$run_name"

[deployment]
type = "single_node"
gpus_per_node = 2
num_gpus = 2

[model]
name = "$source_model"
impl = "custom"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"

[model.vlm]
vision_encoder_attr = "model.visual"
language_model_attr = "model.language_model"
freeze_vision_encoder = true

[model.compile]
fullgraph = false

[model.ac]
mode = "full"
freq = 1
targets = ["norm"]

[tokenizer]
name = "$source_model"

[renderer]
name = "qwen3.5"
enable_thinking = false

[data]
type = "sft"
name = "$corpus_dir"
batch_size = 2
micro_batch_size = 1
seq_len = 16384
shuffle = false
seed = 20260830

[data.loss_mask]
system = false
user = false
assistant = true
tool = false

[optim]
type = "adamw"
lr = $learning_rate
weight_decay = 0.01
max_norm = 1.0
betas1 = 0.9
betas2 = 0.999

[scheduler]
type = "constant"

[ckpt]
interval = 1
keep_last = 1
weights_only = true

[ckpt.weights]
save_sharded = true
save_format = "safetensors"

[file_monitor]
filename = "metrics.jsonl"
EOF

NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 "$uv_bin" run --frozen --no-sync sft @ "$config"
weights=$run_output/weights/step_1
[[ -f "$weights/STABLE" && -f "$weights/model.safetensors" && -f "$run_output/metrics.jsonl" ]] || {
  echo "delayed Designer SFT produced no complete checkpoint" >&2
  exit 1
}
output_sha=$(sha256sum "$weights/model.safetensors" | awk '{print $1}')
[[ "$output_sha" != "$source_sha" ]] || { echo "delayed Designer update did not mutate weights" >&2; exit 1; }
"$uv_bin" run --frozen --no-sync python - \
  "$receipt" "$role" "$source_model" "$source_sha" "$weights" "$output_sha" \
  "$generation" "$score" "$corpus_dir/MANIFEST.json" "$run_output/metrics.jsonl" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

receipt, role, source, source_sha, output, output_sha, generation, score, manifest, metrics = sys.argv[1:]
generation_data = json.load(open(generation))
manifest_data = json.load(open(manifest))
payload = {
    "schema_version": "qwen35-2b-delayed-role-local-designer-update/v1",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "role": role,
    "objective": "environment_designer",
    "training_stage": "one-update-delayed_reward_filtered_coevolution",
    "optimizer_updates": 1,
    "full_dense": True,
    "lora": False,
    "source": {"path": source, "model_sha256": source_sha},
    "output": {"path": output, "model_sha256": output_sha},
    "source_batch_id": generation_data["batch_id"],
    "designer_model_sha256_at_generation": generation_data["designer_model"]["weight_sha256"],
    "selected_environment_ids": manifest_data["selected_environment_ids"],
    "generation": {"path": generation, "sha256": digest(generation)},
    "score": {"path": score, "sha256": digest(score)},
    "corpus": {"path": str(Path(manifest).parent), "manifest_sha256": digest(manifest)},
    "metrics": {"path": metrics, "sha256": digest(metrics)},
}
Path(receipt).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
echo "delayed role-local Designer update completed: $receipt"
