#!/usr/bin/env python3
"""Audit a matched depth-two/depth-three structural recursion pair."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from summarize_q35_2b_recursive_compute_pair_v1 import (
    _task_contract_sha256,
    _task_name,
)
from summarize_q35_2b_topology_replications_v1 import load_traces, selected_topology


def p_boundary_reasons(trace: dict[str, Any]) -> list[str]:
    metrics = trace.get("metrics") or {}
    checks = {
        "topology_hierarchical": metrics.get("topology_hierarchical") == 1,
        "top_manager_admitted": metrics.get("depth3_top_manager_spawns") == 1,
        "both_subgroup_managers_admitted": metrics.get("depth3_subgroup_manager_spawns") == 2,
        "leaf_layer_incomplete": metrics.get("depth3_leaf_spawns", 0) < 3,
        "leaf_admission_rejected": metrics.get("coordination_failed_spawn_calls", 0) >= 1,
        "graph_incomplete": metrics.get("depth3_graph_complete") == 0,
        "hard_failure": not hard_success(trace),
    }
    return [name for name, passed in checks.items() if not passed]


def pplus_structural_reasons(trace: dict[str, Any]) -> list[str]:
    metrics = trace.get("metrics") or {}
    checks = {
        "topology_hierarchical": metrics.get("topology_hierarchical") == 1,
        "topology_utility_aligned": metrics.get("topology_utility_aligned") == 1,
        "exact_d3_graph": metrics.get("depth3_graph_complete") == 1,
        "maximum_depth_three": metrics.get("maximum_exercised_coordination_depth") == 3,
        "six_admissions": metrics.get("coordination_spawn_calls") == 6,
        "no_failed_admissions": metrics.get("coordination_failed_spawn_calls") == 0,
        "fan_in_complete": metrics.get("fan_in_complete") == 1,
        "protocol_aligned": metrics.get("protocol_aligned") == 1,
        "clean_protocol_aligned": metrics.get("clean_protocol_aligned") == 1,
        "no_failed_cells": metrics.get("failed_cells", 0) == 0,
        "no_delegated_path_access": metrics.get("coordinator_delegated_path_accesses", 0) == 0,
        "agent_completed": trace.get("stop_condition") == "agent_completed",
    }
    return [name for name, passed in checks.items() if not passed]


def hard_success(trace: dict[str, Any]) -> bool:
    metrics = trace.get("metrics") or {}
    return not pplus_structural_reasons(trace) and metrics.get("answer_accuracy") == 1


def select_answer_free(attempts: list[dict[str, Any]]) -> int:
    if not attempts:
        raise ValueError("P+ task has no attempts")
    return next(
        (
            index
            for index, trace in enumerate(attempts)
            if not pplus_structural_reasons(trace)
        ),
        0,
    )


def _attempt_row(trace: dict[str, Any], index: int) -> dict[str, Any]:
    metrics = trace.get("metrics") or {}
    reasons = pplus_structural_reasons(trace)
    return {
        "attempt": index + 1,
        "trace_id": trace.get("id"),
        "selected_topology": selected_topology(metrics),
        "structurally_eligible": not reasons,
        "structural_reasons": reasons,
        "answer_accuracy": metrics.get("answer_accuracy"),
        "hard_success": hard_success(trace),
        "coordination_spawn_calls": metrics.get("coordination_spawn_calls"),
        "depth3_graph_complete": metrics.get("depth3_graph_complete"),
        "maximum_exercised_coordination_depth": metrics.get(
            "maximum_exercised_coordination_depth"
        ),
    }


def summarize_pair(
    p_path: Path,
    pplus_path: Path,
    *,
    expected_tasks: int = 8,
    pplus_attempts: int = 3,
    gap_floor: int = 4,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"p": [], "p_plus": []}
    )
    for arm, path in (("p", p_path), ("p_plus", pplus_path)):
        for trace in load_traces(path):
            grouped[_task_name(trace)][arm].append(trace)
    if len(grouped) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} paired tasks, found {len(grouped)}")

    rows = []
    for task_name, arms in sorted(grouped.items()):
        if len(arms["p"]) != 1 or len(arms["p_plus"]) != pplus_attempts:
            raise ValueError(
                f"{task_name}: expected 1 P and {pplus_attempts} P+ attempts, "
                f"found {len(arms['p'])} and {len(arms['p_plus'])}"
            )
        contract_hashes = {
            _task_contract_sha256(trace) for trace in (*arms["p"], *arms["p_plus"])
        }
        if len(contract_hashes) != 1:
            raise ValueError(f"{task_name}: P and P+ task contracts differ")
        p_trace = arms["p"][0]
        p_reasons = p_boundary_reasons(p_trace)
        pplus_index = select_answer_free(arms["p_plus"])
        selected = arms["p_plus"][pplus_index]
        gap = not p_reasons and hard_success(selected)
        rows.append(
            {
                "task": task_name,
                "task_contract_sha256": contract_hashes.pop(),
                "p": {
                    "trace_id": p_trace.get("id"),
                    "boundary_eligible": not p_reasons,
                    "boundary_reasons": p_reasons,
                    "answer_accuracy": (p_trace.get("metrics") or {}).get(
                        "answer_accuracy"
                    ),
                },
                "p_plus": {
                    "selected_attempt": pplus_index + 1,
                    "attempts": [
                        _attempt_row(trace, index)
                        for index, trace in enumerate(arms["p_plus"])
                    ],
                },
                "qualifying_structural_gap": gap,
            }
        )

    gaps = [row["task"] for row in rows if row["qualifying_structural_gap"]]
    return {
        "schema_version": "q35-2b-recursive-depth3-structural-summary/v1",
        "selector_uses_expected_answer": False,
        "task_contracts_identical": True,
        "task_count": len(rows),
        "p_boundary_eligible_count": sum(row["p"]["boundary_eligible"] for row in rows),
        "p_plus_selected_hard_successes": sum(
            row["p_plus"]["attempts"][row["p_plus"]["selected_attempt"] - 1][
                "hard_success"
            ]
            for row in rows
        ),
        "qualifying_structural_gap_count": len(gaps),
        "qualifying_structural_gap_tasks": gaps,
        "gap_floor": gap_floor,
        "gap_floor_relaxed": False,
        "passed": len(gaps) >= gap_floor,
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=Path, required=True)
    parser.add_argument("--p-plus", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=8)
    parser.add_argument("--p-plus-attempts", type=int, default=3)
    parser.add_argument("--gap-floor", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize_pair(
        args.p,
        args.p_plus,
        expected_tasks=args.expected_tasks,
        pplus_attempts=args.p_plus_attempts,
        gap_floor=args.gap_floor,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
