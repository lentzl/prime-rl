#!/usr/bin/env python3
"""Audit fresh document-topology replication banks and report confusion explicitly."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

FAMILY_TO_TOPOLOGY = {
    "document_utility_direct": "direct",
    "document_utility_flat": "flat",
    "document_utility_hierarchical": "hierarchical",
}
TOPOLOGIES = tuple(FAMILY_TO_TOPOLOGY.values())
ZERO_INVARIANTS = (
    "coordinator_delegated_path_accesses",
    "failed_cells",
    "duplicate_cells",
    "post_parent_send_tool_calls",
)


def load_traces(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing trace file: {path}")
    traces: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            nested = value.get("traces") if isinstance(value, dict) else None
            if not isinstance(nested, list):
                raise ValueError(f"{path}:{line_number}: missing trace list")
            traces.extend(trace for trace in nested if isinstance(trace, dict))
    return traces


def selected_topology(metrics: dict[str, Any]) -> str:
    selected = [
        topology
        for topology in TOPOLOGIES
        if metrics.get(f"topology_{topology}") == 1
    ]
    return selected[0] if len(selected) == 1 else "invalid"


def qualification_reasons(trace: dict[str, Any], expected: str, selected: str) -> list[str]:
    metrics = trace.get("metrics") or {}
    reasons = []
    for metric in (
        "answer_accuracy",
        "protocol_aligned",
        "clean_protocol_aligned",
        "topology_valid",
        "topology_utility_aligned",
    ):
        if metrics.get(metric) != 1:
            reasons.append(metric)
    if selected != expected:
        reasons.append("topology_confusion")
    for metric in ZERO_INVARIANTS:
        if metrics.get(metric, 0) != 0:
            reasons.append(metric)
    if expected != "direct" and metrics.get("fan_in_complete") != 1:
        reasons.append("fan_in_complete")
    if trace.get("stop_condition") != "agent_completed":
        reasons.append("stop_condition")
    return reasons


def summarize_bank(
    label: str,
    path: Path,
    *,
    acceptance_floor: int = 4,
    expected_per_family: int = 2,
) -> dict[str, Any]:
    traces = load_traces(path)
    family_counts: Counter[str] = Counter()
    confusion = {
        expected: {selected: 0 for selected in (*TOPOLOGIES, "invalid")}
        for expected in TOPOLOGIES
    }
    qualifiers = 0
    trace_rows = []
    seen_tasks = set()
    for trace in traces:
        task = trace.get("task", {}).get("data", {})
        family = task.get("family")
        expected = FAMILY_TO_TOPOLOGY.get(family)
        if expected is None:
            raise ValueError(f"{label}: unexpected task family: {family!r}")
        task_name = task.get("name")
        if not isinstance(task_name, str) or task_name in seen_tasks:
            raise ValueError(f"{label}: missing or duplicate task name: {task_name!r}")
        seen_tasks.add(task_name)
        family_counts[family] += 1
        metrics = trace.get("metrics") or {}
        selected = selected_topology(metrics)
        confusion[expected][selected] += 1
        reasons = qualification_reasons(trace, expected, selected)
        qualifies = not reasons
        qualifiers += int(qualifies)
        trace_rows.append(
            {
                "task": task_name,
                "family": family,
                "expected": expected,
                "selected": selected,
                "qualifies": qualifies,
                "reasons": reasons,
            }
        )

    expected_counts = {
        family: expected_per_family for family in FAMILY_TO_TOPOLOGY
    }
    balanced = dict(sorted(family_counts.items())) == dict(sorted(expected_counts.items()))
    return {
        "label": label,
        "path": str(path.resolve()),
        "trace_count": len(traces),
        "expected_trace_count": expected_per_family * len(FAMILY_TO_TOPOLOGY),
        "family_counts": dict(sorted(family_counts.items())),
        "balanced": balanced,
        "acceptance_floor": acceptance_floor,
        "acceptance_floor_relaxed": False,
        "qualifying_trajectories": qualifiers,
        "passed": balanced and qualifiers >= acceptance_floor,
        "confusion": confusion,
        "traces": trace_rows,
    }


def summarize_replications(
    banks: list[tuple[str, Path]],
    *,
    acceptance_floor: int = 4,
    expected_per_family: int = 2,
) -> dict[str, Any]:
    if len({label for label, _ in banks}) != len(banks):
        raise ValueError("replication bank labels must be unique")
    summaries = [
        summarize_bank(
            label,
            path,
            acceptance_floor=acceptance_floor,
            expected_per_family=expected_per_family,
        )
        for label, path in banks
    ]
    aggregate_confusion = {
        expected: {selected: 0 for selected in (*TOPOLOGIES, "invalid")}
        for expected in TOPOLOGIES
    }
    for bank in summaries:
        for expected, row in bank["confusion"].items():
            for selected, count in row.items():
                aggregate_confusion[expected][selected] += count
    return {
        "schema_version": "q35-2b-document-topology-replication-summary/v1",
        "acceptance_floor": acceptance_floor,
        "acceptance_floor_relaxed": False,
        "replication_bank_count": len(summaries),
        "passed_bank_count": sum(bank["passed"] for bank in summaries),
        "all_banks_passed": bool(summaries) and all(bank["passed"] for bank in summaries),
        "qualifying_trajectories": sum(
            bank["qualifying_trajectories"] for bank in summaries
        ),
        "aggregate_confusion": aggregate_confusion,
        "banks": summaries,
    }


def parse_bank(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("bank must be LABEL=TRACES_JSONL")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("bank must be LABEL=TRACES_JSONL")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", type=parse_bank, required=True)
    parser.add_argument("--acceptance-floor", type=int, default=4)
    parser.add_argument("--expected-per-family", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.acceptance_floor < 1:
        parser.error("acceptance floor must be positive")
    if args.expected_per_family < 1:
        parser.error("expected per-family count must be positive")
    summary = summarize_replications(
        args.bank,
        acceptance_floor=args.acceptance_floor,
        expected_per_family=args.expected_per_family,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
