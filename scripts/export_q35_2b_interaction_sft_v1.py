#!/usr/bin/env python3
"""Export admitted interaction-curriculum branches into a hash-locked SFT corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from export_prime_agent_role_sft_v1 import _branch_paths, _wire_message, sha256_file
from summarize_q35_2b_interaction_curriculum_v1 import _qualify

SCHEMA_VERSION = "qwen35-2b-interaction-joint-corpus/v2"
BASELINE_SCHEMA_VERSION = "qwen35-2b-interaction-pretraining-baseline/v1"
SUMMARY_SCHEMA_VERSION = "qwen35-2b-interaction-curriculum-summary/v1"
EXPECTED_WEIGHT_SHA = "c75915dd41cd4fc9b1a1ef5582c6fd14913fc6f9971a58feca3b72b4bfcad406"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "") for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def _canonical_paths(trace: dict[str, Any]) -> dict[str, list[int]]:
    nodes = trace["nodes"]
    roots = [index for index, node in enumerate(nodes) if node.get("parent") is None]
    if len(roots) != 2:
        raise ValueError(f"trace {trace.get('id')} must contain one root and one child")
    paths = _branch_paths(trace)
    result: dict[str, list[int]] = {}
    for role, root in (("orchestrator", roots[0]), ("child", roots[1])):
        candidates = [path for path in paths if path[0] == root]
        if not candidates:
            raise ValueError(f"trace {trace.get('id')} lacks a {role} branch")
        longest = max(len(path) for path in candidates)
        selected = [path for path in candidates if len(path) == longest]
        if len(selected) != 1:
            raise ValueError(f"trace {trace.get('id')} has ambiguous {role} branches")
        path = selected[0]
        if not any(nodes[index].get("sampled") is True for index in path):
            raise ValueError(f"trace {trace.get('id')} {role} branch has no sampled turn")
        result[role] = path
    orchestrator_messages = [nodes[index]["message"] for index in result["orchestrator"]]
    if not any(
        message.get("role") == "user" and _content_text(message.get("content")).lstrip().startswith("[from child:")
        for message in orchestrator_messages
    ):
        raise ValueError(f"trace {trace.get('id')} canonical root never receives child")
    return result


def export(
    *,
    traces_paths: list[Path],
    summary_path: Path,
    versions_paths: list[Path],
    output_dir: Path,
    phase: str,
    sampled_model: str,
    student_snapshot: str,
    student_revision: str,
    student_weight_sha: str,
    roles: tuple[str, ...] = ("orchestrator", "child"),
    selection_count: int = 4,
    initial_adapter_path: Path | None = None,
    initial_adapter_sha256: str | None = None,
    dense_weight_mutated: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite corpus: {output_dir}")
    if not dense_weight_mutated and student_weight_sha != EXPECTED_WEIGHT_SHA:
        raise ValueError("unexpected immutable student weight hash")
    if dense_weight_mutated:
        snapshot_path = Path(student_snapshot)
        if (
            not snapshot_path.is_absolute()
            or not (snapshot_path / "STABLE").is_file()
            or sha256_file(snapshot_path / "model.safetensors") != student_weight_sha
        ):
            raise ValueError("mutated dense student path does not match its SHA-256")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported interaction admission summary")
    if summary.get("phase") != phase:
        raise ValueError("interaction phase does not match admission summary")
    if (
        not roles
        or len(set(roles)) != len(roles)
        or not set(roles) <= {"orchestrator", "coordinator", "child"}
        or {"orchestrator", "coordinator"} <= set(roles)
    ):
        raise ValueError("roles must be unique child/coordinator paths; orchestrator aliases coordinator")
    if not 1 <= selection_count <= 4:
        raise ValueError("selection count must be between one training row and four promotion rows")
    gate = summary.get("gate") or {}
    qualifying = summary.get("qualifying") or []
    if (
        gate.get("acceptance_floor_relaxed") is not False
        or len(qualifying) < selection_count
        or summary.get("distinct_qualifying_task_keys", 0) < selection_count
    ):
        raise ValueError("interaction admission gate is not open for the requested selection")
    selected = sorted(
        qualifying,
        key=lambda item: (item["task_key"], item["trace_id"]),
    )[:selection_count]
    accepted = {item["trace_id"]: item["task_key"] for item in selected}
    if len(accepted) != selection_count:
        raise ValueError("admission contains duplicate trace IDs")
    if (initial_adapter_path is None) != (initial_adapter_sha256 is None):
        raise ValueError("initial adapter path and SHA-256 must be set together")
    if (
        initial_adapter_path is not None
        and sha256_file(initial_adapter_path / "adapter_model.safetensors") != initial_adapter_sha256
    ):
        raise ValueError("initial adapter SHA-256 mismatch")

    rows: list[dict[str, Any]] = []
    found: set[str] = set()
    excluded: set[str] = set()
    role_counts: Counter[str] = Counter()
    for traces_path in traces_paths:
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
                if trace.get("agent", {}).get("config", {}).get("model") != sampled_model:
                    raise ValueError(f"trace {trace_id} was not sampled from expected model")
                if trace_id not in accepted:
                    excluded.add(trace_id)
                    continue
                if trace_id in found or trace.get("task", {}).get("key") != accepted[trace_id]:
                    raise ValueError(f"duplicate or mismatched admitted trace: {trace_id}")
                reasons = _qualify(trace, phase)
                if reasons:
                    raise ValueError(f"trace {trace_id} fails {phase} admission: {reasons}")
                scaffold = (trace.get("info") or {}).get("natural_yield_scaffold") or {}
                exact_expected = phase in {
                    "e0_full_actions",
                    "e0b_select_child_value",
                    "e0c_natural_child",
                    "e0c2_natural_child_no_template",
                    "e0c25_inline_evidence",
                    "e0c275_inline_location",
                    "e0c28_inline_only",
                    "e0c29_evidence_available",
                    "e0c3_natural_child_minimal",
                }
                guided_expected = phase == "e0d_guided_yield"
                capped_expected = phase in {
                    "e0d2_capped_yield",
                    "e0d2_capped_yield_exact_child",
                }
                if (
                    scaffold.get("exact_yield_guidance") is not exact_expected
                    or scaffold.get("guided_yield_instruction") is not guided_expected
                    or (scaffold.get("capped_yield_decode") is True) is not capped_expected
                    or (
                        scaffold.get("decode_constraint")
                        != ("vllm_structured_outputs_choice" if exact_expected else None)
                    )
                ):
                    raise ValueError(f"trace {trace_id} has the wrong yield help level")
                paths = _canonical_paths(trace)
                tools = json.dumps(trace.get("tools") or [], sort_keys=True, separators=(",", ":"))
                for role in roles:
                    path = paths["orchestrator" if role == "coordinator" else role]
                    rows.append(
                        {
                            "messages": [_wire_message(trace["nodes"][index]["message"]) for index in path],
                            "tools": tools,
                            "axis": "natural_n1a",
                            "phase": phase,
                            "task_key": accepted[trace_id],
                            "trace_id": trace_id,
                            "role": role,
                        }
                    )
                    role_counts[role] += 1
                found.add(trace_id)
    expected_role_counts = Counter({role: selection_count for role in roles})
    if found != set(accepted) or role_counts != expected_role_counts:
        raise ValueError("source did not yield the requested admitted role rows")

    output_dir.mkdir(parents=True)
    from datasets import Dataset

    parquet_path = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet_path))
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "student": {
            "snapshot": student_snapshot,
            "revision": student_revision,
            "weight_sha256": student_weight_sha,
            "dense_weight_mutated": dense_weight_mutated,
        },
        "admission": {
            "phase": phase,
            "eligible_qualifying_trajectories": len(qualifying),
            "selected_qualifying_trajectories": selection_count,
            "distinct_task_keys": selection_count,
            "acceptance_floor_relaxed": False,
            "traces_sha256": [sha256_file(path) for path in traces_paths],
            "summary_sha256": sha256_file(summary_path),
            "versions_sha256": [sha256_file(path) for path in versions_paths],
        },
    }
    if initial_adapter_path is not None:
        baseline["initial_adapter"] = {
            "path": str(initial_adapter_path.resolve()),
            "adapter_model_sha256": initial_adapter_sha256,
        }
    baseline_path = output_dir / "PRETRAINING-BASELINE.json"
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "student": baseline["student"],
        "rows": len(rows),
        "rows_by_role": dict(sorted(role_counts.items())),
        "selected_roles": list(roles),
        "selection_count": selection_count,
        "accepted_trace_ids": sorted(accepted),
        "accepted_task_keys": sorted(accepted.values()),
        "excluded_trace_ids": sorted(excluded),
        "source": {
            "traces_paths": [str(path.resolve()) for path in traces_paths],
            "traces_sha256": [sha256_file(path) for path in traces_paths],
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "versions_paths": [str(path.resolve()) for path in versions_paths],
            "versions_sha256": [sha256_file(path) for path in versions_paths],
        },
        "dataset": {"path": parquet_path.name, "sha256": sha256_file(parquet_path)},
        "pretraining_baseline": {
            "path": baseline_path.name,
            "sha256": sha256_file(baseline_path),
        },
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--versions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--sampled-model", required=True)
    parser.add_argument("--student-snapshot", required=True)
    parser.add_argument("--student-revision", required=True)
    parser.add_argument("--student-weight-sha", required=True)
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("orchestrator", "coordinator", "child"),
        default=("orchestrator", "child"),
    )
    parser.add_argument("--selection-count", type=int, default=4)
    parser.add_argument("--initial-adapter-path", type=Path)
    parser.add_argument("--initial-adapter-sha256")
    parser.add_argument("--dense-weight-mutated", action="store_true")
    args = parser.parse_args()
    manifest = export(
        traces_paths=args.traces,
        summary_path=args.summary,
        versions_paths=args.versions,
        output_dir=args.output_dir,
        phase=args.phase,
        sampled_model=args.sampled_model,
        student_snapshot=args.student_snapshot,
        student_revision=args.student_revision,
        student_weight_sha=args.student_weight_sha,
        roles=tuple(args.roles),
        selection_count=args.selection_count,
        initial_adapter_path=args.initial_adapter_path,
        initial_adapter_sha256=args.initial_adapter_sha256,
        dense_weight_mutated=args.dense_weight_mutated,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
