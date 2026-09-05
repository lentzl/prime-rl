#!/usr/bin/env python3
"""Build a leak-then-tighten first-call curriculum for the source specialist."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file
from export_q35_2b_specialist_worker_sft_v1 import (
    SCHEMA_VERSION as BASE_SCHEMA_VERSION,
)
from export_q35_2b_specialist_worker_sft_v1 import _assignment, _family_target

SCHEMA_VERSION = "qwen35-2b-source-worker-remedial-sft/v1"
OBJECTIVE = "answer_free_source_specialist_first_call_compute_and_parent_report"
FAMILIES = ("source_ast", "source_config")
PHASES = ("procedure_leak", "live_shape")


def _base_context(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    manifest_path = path / "MANIFEST.json"
    parquet = path / "train.parquet"
    if not manifest_path.is_file() or not parquet.is_file():
        raise FileNotFoundError(f"incomplete base specialist dataset: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != BASE_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("role") != "child"
        or manifest.get("expert_id") != "source_inspector"
        or manifest.get("training_template_variants") != [0, 1, 2, 3]
        or manifest.get("heldout_template_variants_excluded") != [4, 5]
        or manifest.get("answer_free") is not True
        or manifest.get("dataset", {}).get("path") != parquet.name
        or manifest.get("dataset", {}).get("sha256") != sha256_file(parquet)
    ):
        raise ValueError("base source specialist dataset is invalid")
    rows = list(Dataset.from_parquet(str(parquet)))
    if not rows:
        raise ValueError("base source specialist dataset is empty")
    first = rows[0]
    messages = first.get("messages")
    tools = first.get("tools")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        raise ValueError("base source specialist dataset lacks a runtime message")
    if not isinstance(tools, str):
        raise ValueError("base source specialist dataset lacks serialized tools")
    parsed_tools = json.loads(tools)
    if [tool.get("name") for tool in parsed_tools if isinstance(tool, dict)] != ["ipython"]:
        raise ValueError("remedial curriculum requires exactly the live IPython tool")
    return copy.deepcopy(messages[0]), tools, manifest


def _procedure_leak(family: str) -> str:
    common = (
        "[training-only first-call procedure leak]\n"
        "Make the first IPython call the only work call. It must read every assigned file, "
        "compute the exact integer, serialize {'value': value} with "
        "json.dumps(..., separators=(',', ':')), and await "
        "agent_message.send(..., receiver_role='parent') in that same call. Never send a dict "
        "or bare scalar, and do not inspect or retry first.\n"
    )
    if family == "source_ast":
        return common + (
            "For each supplied .py path, execute ast.parse(path.read_text()) and traverse the "
            "result with ast.walk. Count ast.FunctionDef and ast.AsyncFunctionDef separately. "
            "Count decorated function nodes with bool(node.decorator_list), then apply the "
            "requested 2*sync + 3*async + decorated formula."
        )
    if family == "source_config":
        return common + (
            "Parse service.toml with tomllib.loads(Path(...).read_text()). Build the feature "
            "mapping from nonempty KEY=value lines using line.split('=', 1). Read workers and "
            "timeout_seconds below config['runtime'], count values exactly equal to 'true', "
            "then apply workers * timeout_seconds + enabled_count."
        )
    raise ValueError(f"unsupported remedial family: {family}")


def _row(
    *,
    phase: str,
    family: str,
    variant: int,
    instance: int,
    runtime_message: dict[str, Any],
    tools: str,
    base_dataset: Path,
) -> dict[str, Any]:
    root = f"/workspace/specialist-worker/v{variant}-i{instance}"
    objective, code, reasoning = _family_target(family, root)
    assignment = objective
    if phase == "procedure_leak":
        assignment = f"{objective}\n\n{_procedure_leak(family)}"
        reasoning = (
            "Follow the supplied canonical procedure in one correct first call, "
            "including compact JSON transport."
        )
    else:
        reasoning = (
            f"{reasoning} Perform the complete read, computation, serialization, "
            "and parent send in the first call."
        )
    digest = hashlib.sha256(
        f"{phase}:{family}:{variant}:{instance}:{code}".encode()
    ).hexdigest()[:16]
    call_id = f"source-remedial-{digest}"
    return {
        "messages": [
            copy.deepcopy(runtime_message),
            _wire_message(_assignment("source_inspector", assignment)),
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "ipython",
                            "arguments": json.dumps(
                                {"code": code}, separators=(",", ":")
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "message queued",
                "tool_call_id": call_id,
            },
            {
                "role": "assistant",
                "content": "Done.",
                "reasoning_content": (
                    "The compact JSON report was delivered to my parent in the "
                    "first work call, so I stop."
                ),
                "tool_calls": [],
            },
        ],
        "tools": tools,
        "task_key": f"source-remedial-{phase}-{family}-v{variant}-i{instance}",
        "trace_id": f"source-remedial-sft:{digest}",
        "family": f"specialist_{family}",
        "role": "child",
        "expert_id": "source_inspector",
        "objective": OBJECTIVE,
        "training_phase": phase,
        "source_dataset": str(base_dataset.resolve()),
    }


def export(
    *,
    base_dataset_dir: Path,
    output_dir: Path,
    instances_per_variant: int = 8,
    instance_offset: int = 60000,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite remedial corpus: {output_dir}")
    if instances_per_variant < 1:
        raise ValueError("instances_per_variant must be positive")
    runtime_message, tools, base_manifest = _base_context(base_dataset_dir)
    rows = [
        _row(
            phase=phase,
            family=family,
            variant=variant,
            instance=instance_offset + local_index,
            runtime_message=runtime_message,
            tools=tools,
            base_dataset=base_dataset_dir,
        )
        for phase in PHASES
        for variant in range(4)
        for local_index in range(instances_per_variant)
        for family in FAMILIES
    ]
    expected_per_phase = 4 * instances_per_variant * len(FAMILIES)
    expected_rows = len(PHASES) * expected_per_phase
    if len(rows) != expected_rows or len({row["task_key"] for row in rows}) != expected_rows:
        raise ValueError("remedial corpus cardinality is invalid")
    family_counts = {
        family: sum(row["family"] == f"specialist_{family}" for row in rows)
        for family in FAMILIES
    }
    phase_counts = {
        phase: sum(row["training_phase"] == phase for row in rows)
        for phase in PHASES
    }
    if len(set(family_counts.values())) != 1 or phase_counts != {
        phase: expected_per_phase for phase in PHASES
    }:
        raise ValueError("remedial corpus is not family- and phase-balanced")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "child",
        "expert_id": "source_inspector",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": family_counts,
        "phase_counts": phase_counts,
        "curriculum_phase_order": list(PHASES),
        "training_template_variants": [0, 1, 2, 3],
        "instance_offset": instance_offset,
        "instances_per_variant": instances_per_variant,
        "heldout_template_variants_excluded": [4, 5],
        "answer_free": True,
        "heldout_tasks_or_values_used": False,
        "training_only_procedure_leak": True,
        "live_shape_final_phase": True,
        "first_call_single_ipython_target": True,
        "compact_json_parent_report_in_same_call": True,
        "tool_call_format": "openai_function_v1",
        "shuffle_required": False,
        "base_dataset": {
            "path": str(base_dataset_dir.resolve()),
            "manifest_sha256": sha256_file(base_dataset_dir / "MANIFEST.json"),
            "train_parquet_sha256": base_manifest["dataset"]["sha256"],
        },
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instances-per-variant", type=int, default=8)
    parser.add_argument("--instance-offset", type=int, default=60000)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                base_dataset_dir=args.base_dataset_dir.resolve(),
                output_dir=args.output_dir.resolve(),
                instances_per_variant=args.instances_per_variant,
                instance_offset=args.instance_offset,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
