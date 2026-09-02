#!/usr/bin/env python3
"""Run one bounded full-dense update for an isolated terminal specialist."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_specialist_worker_sft_v1 import (
    FAMILIES,
    OBJECTIVE,
    SPECIALISTS,
)
from export_q35_2b_specialist_worker_sft_v1 import (
    SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
)
from run_q35_2b_document_decision_sft_v1 import training_config

SCHEMA_VERSION = "qwen35-2b-specialist-worker-update/v1"


def _write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _validated_dataset(path: Path, expert_id: str) -> dict[str, Any]:
    manifest_path = path / "MANIFEST.json"
    parquet = path / "train.parquet"
    if not manifest_path.is_file() or not parquet.is_file():
        raise FileNotFoundError(f"incomplete specialist dataset: {path}")
    manifest = json.loads(manifest_path.read_text())
    instances = manifest.get("instances_per_variant")
    expected_per_family = 4 * instances if isinstance(instances, int) else None
    expected_counts = {family: expected_per_family for family in FAMILIES.get(expert_id, ())}
    expected_rows = sum(expected_counts.values()) if expected_counts else None
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("role") != "child"
        or manifest.get("expert_id") != expert_id
        or manifest.get("objective") != OBJECTIVE
        or not isinstance(instances, int)
        or instances < 1
        or manifest.get("rows") != expected_rows
        or manifest.get("family_counts") != expected_counts
        or manifest.get("training_template_variants") != [0, 1, 2, 3]
        or manifest.get("heldout_template_variants_excluded") != [4, 5]
        or manifest.get("answer_free") is not True
        or manifest.get("model_authored_file_computation") is not True
        or manifest.get("strict_json_parent_report") is not True
        or manifest.get("tool_call_format") != "openai_function_v1"
        or manifest.get("dataset", {}).get("path") != parquet.name
        or manifest.get("dataset", {}).get("sha256") != sha256_file(parquet)
    ):
        raise ValueError(f"invalid {expert_id} specialist dataset: {path}")
    return manifest


def _metrics(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                result.update(json.loads(line))
    required = ("loss/mean", "loss/nan_count", "optim/grad_norm", "time/step")
    if any(key not in result for key in required) or result["loss/nan_count"] != 0:
        raise ValueError(f"incomplete specialist metrics: {path}")
    for key in ("loss/mean", "optim/grad_norm", "time/step"):
        if not isinstance(result[key], int | float) or not math.isfinite(result[key]):
            raise ValueError(f"non-finite specialist metric {key}: {result[key]!r}")
    return result


def _gpus_idle() -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_model = args.source_model.resolve()
    source_weight = source_model / "model.safetensors"
    if not source_weight.is_file():
        raise FileNotFoundError(source_weight)
    if not (source_model / "STABLE").is_file():
        raise ValueError(f"source checkpoint is not marked stable: {source_model}")
    source_sha = sha256_file(source_weight)
    dataset_dir = args.dataset_dir.resolve()
    dataset = _validated_dataset(dataset_dir, args.expert_id)
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
    complete = output_weight.is_file() and (output_model / "STABLE").is_file() and metrics_path.is_file()
    if not complete:
        if not _gpus_idle():
            raise RuntimeError("GPUs are not idle at the specialist update boundary")
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
        raise ValueError("specialist update did not change the dense weights")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "algorithm": "sft_terminal_specialist_worker_v1",
        "role": "child",
        "expert_id": args.expert_id,
        "optimizer_updates": args.optimizer_updates,
        "full_dense": True,
        "generic_worker_updated": False,
        "coordinator_updated": False,
        "source": {"model_path": str(source_model), "model_sha256": source_sha},
        "dataset": {
            "path": str(dataset_dir),
            "manifest_sha256": sha256_file(dataset_dir / "MANIFEST.json"),
            "train_parquet_sha256": sha256_file(dataset_dir / "train.parquet"),
            "rows": dataset["rows"],
            "family_counts": dataset["family_counts"],
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
        "output": {"model_path": str(output_model.resolve()), "model_sha256": output_sha},
    }
    receipt_path = state_dir / f"{args.run_name}-receipt.json"
    _write_once(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expert-id", choices=SPECIALISTS, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--optimizer-updates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--uv-bin", type=Path, default=Path("/home/ubuntu/.local/bin/uv"))
    args = parser.parse_args()
    if not 0 < args.learning_rate <= 1e-4:
        parser.error("learning rate is outside the bounded range")
    if not 1 <= args.optimizer_updates <= 8:
        parser.error("optimizer update count is outside the bounded range")
    if args.batch_size not in {4, 6, 8, 12, 16}:
        parser.error("unsupported specialist batch size")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
