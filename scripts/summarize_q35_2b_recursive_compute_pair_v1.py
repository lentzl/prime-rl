#!/usr/bin/env python3
"""Audit a same-task P/P+ recursive-compute pair without answer-aware selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from summarize_q35_2b_topology_replications_v1 import load_traces, selected_topology

ONE_METRICS = (
    "protocol_aligned",
    "clean_protocol_aligned",
    "topology_valid",
    "topology_utility_aligned",
    "fan_in_complete",
)
ZERO_METRICS = (
    "coordinator_delegated_path_accesses",
    "failed_cells",
    "duplicate_cells",
    "post_parent_send_tool_calls",
)


def _task_data(trace: dict[str, Any]) -> dict[str, Any]:
    data = trace.get("task", {}).get("data")
    if not isinstance(data, dict):
        raise ValueError("trace lacks task data")
    return data


def _task_name(trace: dict[str, Any]) -> str:
    name = _task_data(trace).get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("trace lacks a task name")
    return name


def _task_contract_sha256(trace: dict[str, Any]) -> str:
    data = _task_data(trace)
    contract = {
        "name": data.get("name"),
        "family": data.get("family"),
        "prompt": data.get("prompt"),
        "system_prompt": data.get("system_prompt"),
        "files": data.get("files"),
        "answer": data.get("answer"),
        "child_paths": data.get("child_paths"),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def structural_reasons(trace: dict[str, Any]) -> list[str]:
    metrics = trace.get("metrics") or {}
    reasons = [metric for metric in ONE_METRICS if metrics.get(metric) != 1]
    reasons.extend(metric for metric in ZERO_METRICS if metrics.get(metric, 0) != 0)
    if trace.get("stop_condition") != "agent_completed":
        reasons.append("stop_condition")
    return reasons


def hard_success(trace: dict[str, Any]) -> bool:
    metrics = trace.get("metrics") or {}
    return not structural_reasons(trace) and metrics.get("answer_accuracy") == 1


def select_answer_free(attempts: list[dict[str, Any]]) -> int:
    if not attempts:
        raise ValueError("P+ task has no attempts")
    return next(
        (index for index, trace in enumerate(attempts) if not structural_reasons(trace)),
        0,
    )


def _attempt_row(trace: dict[str, Any], index: int) -> dict[str, Any]:
    metrics = trace.get("metrics") or {}
    reasons = structural_reasons(trace)
    return {
        "attempt": index + 1,
        "trace_id": trace.get("id"),
        "selected_topology": selected_topology(metrics),
        "structurally_eligible": not reasons,
        "structural_reasons": reasons,
        "answer_accuracy": metrics.get("answer_accuracy"),
        "hard_success": hard_success(trace),
        "coordination_spawn_calls": metrics.get("coordination_spawn_calls"),
    }


def summarize_pair(
    p_path: Path,
    pplus_path: Path,
    *,
    expected_tasks: int = 12,
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
        pplus_index = select_answer_free(arms["p_plus"])
        pplus_trace = arms["p_plus"][pplus_index]
        p_success = hard_success(p_trace)
        pplus_success = hard_success(pplus_trace)
        rows.append(
            {
                "task": task_name,
                "task_contract_sha256": contract_hashes.pop(),
                "p": _attempt_row(p_trace, 0),
                "p_plus": {
                    "selected_attempt": pplus_index + 1,
                    "attempts": [
                        _attempt_row(trace, index)
                        for index, trace in enumerate(arms["p_plus"])
                    ],
                },
                "compute_gap": not p_success and pplus_success,
            }
        )

    gaps = [row["task"] for row in rows if row["compute_gap"]]
    return {
        "schema_version": "q35-2b-recursive-compute-p-pplus-summary/v1",
        "selector_uses_expected_answer": False,
        "task_contracts_identical": True,
        "task_count": len(rows),
        "p_hard_successes": sum(row["p"]["hard_success"] for row in rows),
        "p_plus_selected_hard_successes": sum(
            row["p_plus"]["attempts"][row["p_plus"]["selected_attempt"] - 1][
                "hard_success"
            ]
            for row in rows
        ),
        "compute_gap_count": len(gaps),
        "compute_gap_tasks": gaps,
        "gap_floor": gap_floor,
        "gap_floor_relaxed": False,
        "passed": len(gaps) >= gap_floor,
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=Path, required=True)
    parser.add_argument("--p-plus", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=12)
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
