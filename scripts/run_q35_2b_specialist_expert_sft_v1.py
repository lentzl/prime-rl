#!/usr/bin/env python3
"""Run a bounded full-dense expert-only update from protected e33."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_specialist_expert_sft_v1 import (
    EXPERT_IDS,
    OBJECTIVE,
    ROWS,
    ROWS_PER_EXPERT,
)
from export_q35_2b_specialist_expert_sft_v1 import (
    SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
)
from run_q35_2b_document_decision_sft_v1 import training_config
from run_q35_2b_specialist_worker_sft_v1 import _gpus_idle, _metrics, _write_once

SCHEMA_VERSION = "qwen35-2b-specialist-expert-update/v2"


def _validated_dataset(path: Path) -> dict[str, Any]:
    manifest_path = path / "MANIFEST.json"
    parquet = path / "train.parquet"
    if not manifest_path.is_file() or not parquet.is_file():
        raise FileNotFoundError(f"incomplete specialist expert dataset: {path}")
    manifest = json.loads(manifest_path.read_text())
    expected_counts = {expert_id: ROWS_PER_EXPERT for expert_id in EXPERT_IDS}
    expected_roles = {
        "generic_worker": {"root": 16, "nonroot_specialist_manager": 0},
        "table_analyst": {"root": 8, "nonroot_specialist_manager": 8},
        "source_inspector": {"root": 8, "nonroot_specialist_manager": 8},
    }
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("role") != "coordinator"
        or manifest.get("objective") != OBJECTIVE
        or manifest.get("rows") != ROWS
        or manifest.get("training_batch_size") != 12
        or manifest.get("expert_counts") != expected_counts
        or manifest.get("role_counts") != expected_roles
        or manifest.get("first_batch_expert_counts")
        != {expert_id: 4 for expert_id in EXPERT_IDS}
        or manifest.get("training_instance_offset") != 37600
        or manifest.get("training_template_variants") != [0, 1, 2, 3]
        or manifest.get("heldout_template_variants_excluded") != [4, 5]
        or manifest.get("observed_instance_offsets_excluded")
        != [35100, 37100, 37200, 37300]
        or manifest.get("answer_free") is not True
        or manifest.get("public_capability_registry_only") is not True
        or manifest.get("expert_only_tool_arguments") is not True
        or manifest.get("cognitive_action_labels_present") is not False
        or manifest.get("root_and_nonroot_coordinator_rows") is not True
        or manifest.get("tool_call_format") != "openai_function_v1"
        or manifest.get("dataset", {}).get("path") != parquet.name
        or manifest.get("dataset", {}).get("sha256") != sha256_file(parquet)
    ):
        raise ValueError(f"invalid specialist expert dataset: {path}")
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_model = args.source_model.resolve()
    source_weight = source_model / "model.safetensors"
    if not source_weight.is_file():
        raise FileNotFoundError(source_weight)
    if not (source_model / "STABLE").is_file():
        raise ValueError(f"source checkpoint is not marked stable: {source_model}")
    source_sha = sha256_file(source_weight)
    if source_sha != args.expected_source_sha256:
        raise ValueError("expert-only parent hash differs from protected e33")

    dataset_dir = args.dataset_dir.resolve()
    dataset = _validated_dataset(dataset_dir)
    output_root = args.output_root.resolve()
    state_dir = args.state_dir.resolve()
    config_path = state_dir / f"{args.run_name}.toml"
    config = training_config(
        run_name=args.run_name,
        model_path=source_model,
        dataset_dir=dataset_dir,
        output_root=output_root,
        learning_rate=args.learning_rate,
        optimizer_updates=args.optimizer_updates,
        batch_size=args.batch_size,
    )
    _write_once(config_path, config)

    output_dir = output_root / args.run_name
    output_model = output_dir / f"weights/step_{args.optimizer_updates}"
    output_weight = output_model / "model.safetensors"
    metrics_path = output_dir / "metrics.jsonl"
    complete = (
        output_weight.is_file()
        and (output_model / "STABLE").is_file()
        and metrics_path.is_file()
    )
    if not complete:
        if not _gpus_idle():
            raise RuntimeError("GPUs are not idle at the expert-only update boundary")
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
    if output_sha == source_sha:
        raise ValueError("expert-only update did not change the dense weights")

    policy_role = "specialist_router" if args.isolated_router else "coordinator"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "algorithm": "sft_public_registry_expert_only_v1",
        "role": policy_role,
        "isolated_router_policy": args.isolated_router,
        "optimizer_updates": args.optimizer_updates,
        "full_dense": True,
        "cognitive_action_trained": False,
        "generic_worker_updated": False,
        "specialist_workers_updated": False,
        "source": {"model_path": str(source_model), "model_sha256": source_sha},
        "dataset": {
            "path": str(dataset_dir),
            "manifest_sha256": sha256_file(dataset_dir / "MANIFEST.json"),
            "train_parquet_sha256": sha256_file(dataset_dir / "train.parquet"),
            "rows": dataset["rows"],
            "expert_counts": dataset["expert_counts"],
            "role_counts": dataset["role_counts"],
            "first_batch_expert_counts": dataset["first_batch_expert_counts"],
            "answer_free": True,
        },
        "training": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "loss_mean": metrics["loss/mean"],
            "loss_nan_count": metrics["loss/nan_count"],
            "gradient_norm": metrics["optim/grad_norm"],
            "time_per_step_seconds": metrics["time/step"],
        },
        "output": {
            "model_path": str(output_model.resolve()),
            "model_sha256": output_sha,
        },
    }
    receipt_path = state_dir / f"{args.run_name}-receipt.json"
    _write_once(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument(
        "--expected-source-sha256",
        default="e33bd4cdbfd92eb22844dbbde2764aa7fa00e1cd25ca7045f91ce22210499e47",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--optimizer-updates", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--isolated-router", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--uv-bin", type=Path, default=Path("/home/ubuntu/.local/bin/uv")
    )
    args = parser.parse_args()
    if not 0 < args.learning_rate <= 2e-6:
        parser.error("learning rate is outside the preregistered bounded range")
    allowed_updates = {4, 8} if args.isolated_router else {1, 2}
    if args.optimizer_updates not in allowed_updates:
        parser.error(
            "expert-only update count is outside the preregistered policy-role ladder"
        )
    if args.batch_size != 12:
        parser.error("expert-only batch size must remain 12")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
