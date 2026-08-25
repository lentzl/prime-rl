#!/usr/bin/env python3
"""Export hard-safety-validated positive prefixes from incomplete trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_prime_agent_role_sft_v1 import _wire_message, sha256_file
from summarize_q35_2b_interaction_curriculum_v1 import SCHEMA_VERSION as SUMMARY_SCHEMA_VERSION
from summarize_q35_2b_interaction_curriculum_v1 import (
    _qualify,
    positive_prefix_audit,
)

SCHEMA_VERSION = "qwen35-2b-positive-prefix-corpus/v1"


def _path_to_node(nodes: list[dict[str, Any]], target: int) -> list[int]:
    path = []
    seen = set()
    current: int | None = target
    while current is not None:
        if current in seen or not 0 <= current < len(nodes):
            raise ValueError("positive-prefix target has an invalid lineage")
        seen.add(current)
        path.append(current)
        parent = nodes[current].get("parent")
        if parent is not None and not isinstance(parent, int):
            raise ValueError("positive-prefix lineage has a non-integer parent")
        current = parent
    return list(reversed(path))


def export_positive_prefixes(
    *,
    traces_path: Path,
    summary_path: Path,
    output_dir: Path,
    phase: str,
    sampled_model: str,
    student_snapshot: Path,
    student_revision: str,
    student_weight_sha: str,
    role: str,
    max_rows: int = 4,
) -> dict[str, Any]:
    if role not in {"coordinator", "child"}:
        raise ValueError("positive-prefix role must be coordinator or child")
    if not 1 <= max_rows <= 4:
        raise ValueError("positive-prefix row cap must be between one and four")
    if output_dir.exists():
        manifest = output_dir / "MANIFEST.json"
        skipped = output_dir / "SKIPPED.json"
        if manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8"))
        if skipped.is_file():
            return json.loads(skipped.read_text(encoding="utf-8"))
        raise ValueError(f"refusing to reuse partial positive-prefix corpus: {output_dir}")
    if (
        not student_snapshot.is_absolute()
        or not (student_snapshot / "STABLE").is_file()
        or sha256_file(student_snapshot / "model.safetensors") != student_weight_sha
    ):
        raise ValueError("positive-prefix student checkpoint is incomplete or has the wrong hash")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION or summary.get("phase") != phase:
        raise ValueError("positive-prefix summary schema or phase mismatch")
    summary_has_prefix_audit = "positive_prefixes" in summary
    admitted = {
        item["trace_id"]: item
        for item in summary.get("positive_prefixes") or []
        if role in (item.get("roles") or {})
    }
    candidates = []
    with traces_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            envelope = json.loads(line)
            traces = envelope.get("traces")
            if not isinstance(traces, list) or len(traces) != 1:
                raise ValueError(f"{traces_path}:{line_number} is not one trace envelope")
            trace = traces[0]
            trace_id = trace.get("id")
            reasons = _qualify(trace, phase)
            if not reasons:
                continue
            audit = positive_prefix_audit(trace, phase)
            if not summary_has_prefix_audit and role in audit:
                task_data = trace.get("task", {}).get("data", {})
                admitted[trace_id] = {
                    "task_key": task_data.get("episode_id") or task_data.get("name"),
                    "trace_id": trace_id,
                    "roles": audit,
                }
            if trace_id not in admitted:
                continue
            if trace.get("agent", {}).get("config", {}).get("model") != sampled_model:
                raise ValueError(f"trace {trace_id} was not sampled from the expected role pair")
            role_audit = audit.get(role)
            if role_audit != admitted[trace_id]["roles"][role]:
                raise ValueError(f"trace {trace_id} positive-prefix audit changed after summarization")
            nodes = trace.get("nodes")
            if not isinstance(nodes, list):
                raise ValueError(f"trace {trace_id} lacks message nodes")
            path = _path_to_node(nodes, role_audit["target_node_index"])
            task_key = trace.get("task", {}).get("data", {}).get("episode_id") or trace.get("task", {}).get(
                "data", {}
            ).get("name")
            if task_key != admitted[trace_id]["task_key"]:
                raise ValueError(f"trace {trace_id} task key changed after summarization")
            candidates.append(
                {
                    "messages": [_wire_message(nodes[index]["message"]) for index in path],
                    "tools": json.dumps(trace.get("tools") or [], sort_keys=True, separators=(",", ":")),
                    "axis": "natural_n1a",
                    "phase": phase,
                    "task_key": task_key,
                    "trace_id": f"positive-prefix:{role}:{trace_id}",
                    "role": role,
                    "objective": "interaction_positive_prefix",
                    "validated_atoms": role_audit["atoms"],
                    "source_trace_id": trace_id,
                }
            )
    candidates.sort(key=lambda row: (-len(row["validated_atoms"]), row["task_key"], row["source_trace_id"]))
    rows = candidates[:max_rows]
    output_dir.mkdir(parents=True)
    if not rows:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "no_hard_safety_validated_positive_prefixes",
            "role": role,
            "rows": 0,
            "source_summary_sha256": sha256_file(summary_path),
        }
        (output_dir / "SKIPPED.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "training_stage": "hard_safety_validated_positive_prefix",
        "hard_safety_validated": True,
        "role": role,
        "selected_roles": [role],
        "rows": len(rows),
        "rows_by_role": {role: len(rows)},
        "student": {
            "snapshot": str(student_snapshot.resolve()),
            "revision": student_revision,
            "weight_sha256": student_weight_sha,
            "dense_weight_mutated": True,
        },
        "accepted_trace_ids": [row["source_trace_id"] for row in rows],
        "validated_atoms": {
            row["source_trace_id"]: row["validated_atoms"] for row in rows
        },
        "source": {
            "traces_path": str(traces_path.resolve()),
            "traces_sha256": sha256_file(traces_path),
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
        },
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--sampled-model", required=True)
    parser.add_argument("--student-snapshot", type=Path, required=True)
    parser.add_argument("--student-revision", required=True)
    parser.add_argument("--student-weight-sha", required=True)
    parser.add_argument("--role", choices=("coordinator", "child"), required=True)
    parser.add_argument("--max-rows", type=int, default=4)
    args = parser.parse_args()
    result = export_positive_prefixes(
        traces_path=args.traces.resolve(),
        summary_path=args.summary.resolve(),
        output_dir=args.output_dir.resolve(),
        phase=args.phase,
        sampled_model=args.sampled_model,
        student_snapshot=args.student_snapshot.resolve(),
        student_revision=args.student_revision,
        student_weight_sha=args.student_weight_sha,
        role=args.role,
        max_rows=args.max_rows,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
