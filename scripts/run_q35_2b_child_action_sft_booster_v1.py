#!/usr/bin/env python3
"""Insert one bounded full-dense child-action SFT update into a role-GRPO journal."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from export_prime_agent_role_sft_v1 import sha256_file
from run_q35_2b_role_grpo_autonomous_v1 import (
    PROMOTION_MINIMUM,
    _candidate,
    _gpus_idle,
    append_event,
    load_events,
    project,
)

SCHEMA_VERSION = "qwen35-2b-child-action-booster-update/v1"
DATASET_SCHEMA_VERSION = "qwen35-2b-child-action-booster-sft/v1"
DATASET_SCHEMA_VERSIONS = {
    DATASET_SCHEMA_VERSION,
    "qwen35-2b-child-action-booster-sft/v2",
}
DATASET_OBJECTIVES = {
    "canonical_exact_parent_send_then_stop",
    "canonical_exact_parent_send_ack_then_stop",
}


def _quote(value: str | Path) -> str:
    return json.dumps(str(value))


def training_config(
    *,
    run_name: str,
    model_path: Path,
    dataset_dir: Path,
    output_root: Path,
    rows: int,
    lr: float,
    optimizer_updates: int,
) -> str:
    if not 2 <= rows <= 32:
        raise ValueError("child-action booster requires between two and thirty-two rows")
    if not 1 <= optimizer_updates <= 8:
        raise ValueError("child-action booster requires between one and eight updates")
    batch_size = rows + rows % 2
    return f"""max_steps = {optimizer_updates}
output_dir = {_quote(output_root)}
clean = false

[run]
name = {_quote(run_name)}
dir = {_quote(run_name)}

[deployment]
type = "single_node"
gpus_per_node = 2
num_gpus = 2

[model]
name = {_quote(model_path)}
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
name = {_quote(model_path)}

[renderer]
name = "qwen3.5"
enable_thinking = false

[data]
type = "sft"
name = {_quote(dataset_dir)}
batch_size = {batch_size}
micro_batch_size = 1
seq_len = 16384
shuffle = false
seed = 20260827

[data.loss_mask]
system = false
user = false
assistant = true
tool = false

[optim]
type = "adamw"
lr = {lr:.12g}
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
"""


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"existing file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _metrics(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.update(json.loads(line))
    required = ("loss/mean", "loss/nan_count", "optim/grad_norm", "time/step")
    if any(key not in result for key in required) or result["loss/nan_count"] != 0:
        raise ValueError(f"incomplete or non-finite booster metrics: {path}")
    return result


def _validated_dataset(path: Path) -> dict[str, Any]:
    manifest_path = path / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    parquet = path / "train.parquet"
    if (
        manifest.get("schema_version") not in DATASET_SCHEMA_VERSIONS
        or manifest.get("status") != "complete"
        or manifest.get("role") != "child"
        or manifest.get("objective") not in DATASET_OBJECTIVES
        or not 2 <= manifest.get("rows", 0) <= 32
        or manifest.get("dataset", {}).get("path") != parquet.name
        or manifest.get("dataset", {}).get("sha256") != sha256_file(parquet)
    ):
        raise ValueError(f"invalid child-action booster dataset: {path}")
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = args.state_dir.resolve()
    events_path = state_dir / "events.jsonl"
    events = load_events(events_path)
    state = project(events)
    if state["pending_eval"] is not None:
        raise ValueError("cannot insert booster while an admission is pending")
    if state["next_cycle"] != args.cycle or state["next_role"] != "coordinator":
        raise ValueError(
            f"booster boundary mismatch: expected coordinator cycle {args.cycle}, "
            f"got {state['next_role']} cycle {state['next_cycle']}"
        )
    cycle_events = [event for event in events if event.get("cycle") == args.cycle]
    started = next((event for event in cycle_events if event["kind"] == "train_started"), None)
    completed = next((event for event in cycle_events if event["kind"] == "train_completed"), None)
    if completed is not None:
        if completed.get("run_name") != args.run_name:
            raise ValueError(f"cycle {args.cycle} is already completed by another run")
        return completed
    if started is not None and started.get("run_name") != args.run_name:
        raise ValueError(f"cycle {args.cycle} is already owned by another run")

    dataset_dir = args.dataset_dir.resolve()
    dataset = _validated_dataset(dataset_dir)
    source = state["frontier"]["child"]
    anchor = state["frontier"]["coordinator"]
    source_candidate = _candidate(Path(source["model_path"]), source["label"])
    anchor_candidate = _candidate(Path(anchor["model_path"]), anchor["label"])
    if source_candidate["model_sha256"] != source["model_sha256"]:
        raise ValueError("child frontier hash changed before booster")
    if anchor_candidate["model_sha256"] != anchor["model_sha256"]:
        raise ValueError("coordinator frontier hash changed before booster")

    phase = state["phases"]["child"]
    leak = state["leak_levels"]["child"]
    if started is None:
        append_event(
            events_path,
            {
                "kind": "train_started",
                "cycle": args.cycle,
                "role": "child",
                "phase": phase,
                "bootstrap_leak_level": leak,
                "run_name": args.run_name,
                "algorithm": "sft_child_action_booster_v1",
                "source": source,
                "anchor": anchor,
                "dataset_dir": str(dataset_dir),
                "dataset_manifest_sha256": sha256_file(dataset_dir / "MANIFEST.json"),
                "runtime_container_baseline": [],
            },
        )

    output_root = args.output_root.resolve()
    config_path = state_dir / f"{args.run_name}.toml"
    config = training_config(
        run_name=args.run_name,
        model_path=Path(source["model_path"]),
        dataset_dir=dataset_dir,
        output_root=output_root,
        rows=dataset["rows"],
        lr=args.learning_rate,
        optimizer_updates=args.optimizer_updates,
    )
    _write_once(config_path, config)
    output_dir = output_root / args.run_name
    output_model = output_dir / f"weights/step_{args.optimizer_updates}"
    output_weight = output_model / "model.safetensors"
    metrics_path = output_dir / "metrics.jsonl"
    if not (output_weight.is_file() and (output_model / "STABLE").is_file() and metrics_path.is_file()):
        if not _gpus_idle():
            raise RuntimeError("GPUs are not idle at the child booster boundary")
        env = os.environ.copy()
        env.update({"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0"})
        subprocess.run(
            [str(args.uv_bin), "run", "--no-sync", "sft", "@", str(config_path)],
            cwd=args.repo.resolve(),
            env=env,
            check=True,
            timeout=args.timeout,
        )
    metrics = _metrics(metrics_path)
    output_sha = sha256_file(output_weight)
    if output_sha == source["model_sha256"]:
        raise ValueError("child booster did not change the dense weights")
    output = _candidate(output_model.resolve(), args.run_name)
    if output["model_sha256"] != output_sha:
        raise ValueError("child booster output hash verification failed")

    receipt_path = state_dir / f"{args.run_name}-receipt.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "algorithm": "sft_child_action_booster_v1",
        "role": "child",
        "phase": phase,
        "bootstrap_leak_level": leak,
        "cycle": args.cycle,
        "optimizer_updates": args.optimizer_updates,
        "full_dense": True,
        "promotion_minimum": PROMOTION_MINIMUM,
        "source": source,
        "anchor": anchor,
        "dataset": {
            "path": str(dataset_dir),
            "manifest_sha256": sha256_file(dataset_dir / "MANIFEST.json"),
            "train_parquet_sha256": sha256_file(dataset_dir / "train.parquet"),
            "rows": dataset["rows"],
            "task_keys": dataset["task_keys"],
        },
        "training": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "learning_rate": args.learning_rate,
            "loss_mean": metrics["loss/mean"],
            "loss_nan_count": metrics["loss/nan_count"],
            "gradient_norm": metrics["optim/grad_norm"],
            "time_per_step_seconds": metrics["time/step"],
        },
        "output": output,
    }
    _write_once(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    event = append_event(
        events_path,
        {
            "kind": "train_completed",
            "cycle": args.cycle,
            "role": "child",
            "phase": phase,
            "bootstrap_leak_level": leak,
            "run_name": args.run_name,
            "algorithm": "sft_child_action_booster_v1",
            "receipt_path": str(receipt_path),
            "output_candidate": output,
        },
    )
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--optimizer-updates", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--uv-bin", type=Path, default=Path("/home/ubuntu/.local/bin/uv"))
    args = parser.parse_args()
    if args.cycle < 1 or not 0 < args.learning_rate <= 1e-4 or not 1 <= args.optimizer_updates <= 8:
        parser.error("cycle, learning rate, or update count is outside the bounded range")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
