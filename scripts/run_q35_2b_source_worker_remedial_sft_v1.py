#!/usr/bin/env python3
"""Run one bounded full-dense source-worker remedial curriculum update."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from datasets import Dataset

from export_q35_2b_document_decision_sft_v1 import sha256_file
from export_q35_2b_source_worker_remedial_sft_v1 import (
    FAMILIES,
    OBJECTIVE,
    PHASES,
    SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
)
from run_q35_2b_document_decision_sft_v1 import training_config

SCHEMA_VERSION = "qwen35-2b-source-worker-remedial-update/v1"


def _write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _validated_dataset(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_parquet_sha256: str,
) -> dict[str, Any]:
    manifest_path = path / "MANIFEST.json"
    parquet = path / "train.parquet"
    if not manifest_path.is_file() or not parquet.is_file():
        raise FileNotFoundError(f"incomplete source-worker remedial dataset: {path}")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("source-worker remedial manifest hash mismatch")
    if sha256_file(parquet) != expected_parquet_sha256:
        raise ValueError("source-worker remedial parquet hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = manifest.get("instances_per_variant")
    expected_per_phase = (
        4 * instances * len(FAMILIES) if isinstance(instances, int) else None
    )
    expected_rows = (
        len(PHASES) * expected_per_phase
        if isinstance(expected_per_phase, int)
        else None
    )
    expected_family_count = (
        len(PHASES) * 4 * instances if isinstance(instances, int) else None
    )
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("role") != "child"
        or manifest.get("expert_id") != "source_inspector"
        or manifest.get("objective") != OBJECTIVE
        or instances != 8
        or manifest.get("instance_offset") != 60000
        or manifest.get("rows") != expected_rows
        or manifest.get("family_counts")
        != {family: expected_family_count for family in FAMILIES}
        or manifest.get("phase_counts")
        != {phase: expected_per_phase for phase in PHASES}
        or manifest.get("curriculum_phase_order") != list(PHASES)
        or manifest.get("training_template_variants") != [0, 1, 2, 3]
        or manifest.get("heldout_template_variants_excluded") != [4, 5]
        or manifest.get("answer_free") is not True
        or manifest.get("heldout_tasks_or_values_used") is not False
        or manifest.get("training_only_procedure_leak") is not True
        or manifest.get("live_shape_final_phase") is not True
        or manifest.get("first_call_single_ipython_target") is not True
        or manifest.get("compact_json_parent_report_in_same_call") is not True
        or manifest.get("tool_call_format") != "openai_function_v1"
        or manifest.get("shuffle_required") is not False
        or manifest.get("dataset", {}).get("path") != parquet.name
        or manifest.get("dataset", {}).get("sha256") != expected_parquet_sha256
    ):
        raise ValueError("invalid source-worker remedial dataset")
    rows = list(Dataset.from_parquet(str(parquet)))
    if len(rows) != expected_rows:
        raise ValueError("invalid source-worker remedial row count")
    expected_phase_sequence = [
        *(["procedure_leak"] * expected_per_phase),
        *(["live_shape"] * expected_per_phase),
    ]
    for row, expected_phase in zip(rows, expected_phase_sequence, strict=True):
        messages = row.get("messages")
        if (
            row.get("training_phase") != expected_phase
            or row.get("role") != "child"
            or row.get("expert_id") != "source_inspector"
            or row.get("objective") != OBJECTIVE
            or "answer" in row
            or not isinstance(messages, list)
            or len(messages) != 5
            or not isinstance(messages[1].get("content"), str)
            or len(messages[2].get("tool_calls") or []) != 1
        ):
            raise ValueError("invalid source-worker remedial row contract")
        leak_visible = "[training-only first-call procedure leak]" in messages[1]["content"]
        if leak_visible != (expected_phase == "procedure_leak"):
            raise ValueError("source-worker remedial phase ordering is invalid")
        tool_call = messages[2]["tool_calls"][0]
        function = tool_call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments", ""))
        except json.JSONDecodeError as error:
            raise ValueError("invalid remedial IPython arguments") from error
        code = arguments.get("code") if isinstance(arguments, dict) else None
        family = row.get("family")
        common_target = (
            isinstance(code, str)
            and "read_text" in code
            and "json.dumps({'value': value}" in code
            and "agent_message.send" in code
            and "receiver_role='parent'" in code
        )
        family_target = (
            family == "specialist_source_ast"
            and isinstance(code, str)
            and all(
                token in code
                for token in (
                    "ast.walk(ast.parse(path.read_text()))",
                    "ast.FunctionDef",
                    "ast.AsyncFunctionDef",
                    "bool(node.decorator_list)",
                )
            )
        ) or (
            family == "specialist_source_config"
            and isinstance(code, str)
            and all(
                token in code
                for token in (
                    "tomllib.loads",
                    "line.split('=', 1)",
                    "config['runtime']['workers']",
                    "config['runtime']['timeout_seconds']",
                    "value == 'true'",
                )
            )
        )
        if not common_target or not family_target:
            raise ValueError("source-worker remedial target is not atomic and canonical")
    return manifest


def _metrics(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                result.update(json.loads(line))
    required = ("loss/mean", "loss/nan_count", "optim/grad_norm", "time/step")
    if any(key not in result for key in required) or result["loss/nan_count"] != 0:
        raise ValueError(f"incomplete source-worker remedial metrics: {path}")
    for key in ("loss/mean", "optim/grad_norm", "time/step"):
        if not isinstance(result[key], int | float) or not math.isfinite(result[key]):
            raise ValueError(f"non-finite source-worker remedial metric {key}: {result[key]!r}")
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
    if not source_weight.is_file() or not (source_model / "STABLE").is_file():
        raise ValueError(f"source-worker experimental parent is incomplete: {source_model}")
    source_sha = sha256_file(source_weight)
    if source_sha != args.expected_source_sha256:
        raise ValueError("source-worker experimental parent hash mismatch")
    dataset_dir = args.dataset_dir.resolve()
    dataset = _validated_dataset(
        dataset_dir,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_parquet_sha256=args.expected_parquet_sha256,
    )
    output_root = args.output_root.resolve()
    state_dir = args.state_dir.resolve()
    config_path = state_dir / f"{args.run_name}.toml"
    config = training_config(
        run_name=args.run_name,
        model_path=source_model,
        dataset_dir=dataset_dir,
        output_root=output_root,
        learning_rate=2e-6,
        optimizer_updates=8,
        batch_size=16,
    )
    _write_once(config_path, config)

    output_dir = output_root / args.run_name
    output_model = output_dir / "weights/step_8"
    output_weight = output_model / "model.safetensors"
    metrics_path = output_dir / "metrics.jsonl"
    complete = (
        output_weight.is_file()
        and (output_model / "STABLE").is_file()
        and metrics_path.is_file()
    )
    if not complete:
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to resume or overwrite partial remedial update: {output_dir}"
            )
        if not _gpus_idle():
            raise RuntimeError("GPUs are not idle at the remedial update boundary")
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
        raise ValueError("source-worker remedial update did not change the dense weights")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "algorithm": "sft_source_worker_remedial_leak_then_tighten_v1",
        "role": "child",
        "expert_id": "source_inspector",
        "optimizer_updates": 8,
        "full_dense": True,
        "generic_worker_updated": False,
        "coordinator_updated": False,
        "optimizer_state_continuous_from_source": False,
        "optimizer_state_boundary": "fresh Adam state from exact experimental parent weights",
        "source": {"model_path": str(source_model), "model_sha256": source_sha},
        "dataset": {
            "path": str(dataset_dir),
            "manifest_sha256": args.expected_manifest_sha256,
            "train_parquet_sha256": args.expected_parquet_sha256,
            "rows": dataset["rows"],
            "family_counts": dataset["family_counts"],
            "phase_counts": dataset["phase_counts"],
            "curriculum_phase_order": dataset["curriculum_phase_order"],
            "answer_free": True,
            "heldout_tasks_or_values_used": False,
        },
        "training": {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "learning_rate": 2e-6,
            "batch_size": 16,
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
    _write_once(
        receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-parquet-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--uv-bin", type=Path, default=Path("/home/ubuntu/.local/bin/uv"))
    args = parser.parse_args()
    for label, digest in (
        ("source", args.expected_source_sha256),
        ("manifest", args.expected_manifest_sha256),
        ("parquet", args.expected_parquet_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            parser.error(f"expected {label} SHA-256 is invalid")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
