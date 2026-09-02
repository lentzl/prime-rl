#!/usr/bin/env python3
"""Audit exact public-registry routing on the frozen specialist screen."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from summarize_q35_2b_specialist_population_v1 import (
    SELECTED_EXPERT,
    _audit_terminal_routes,
    _load_jsonl,
    _load_traces,
    _task_data,
    _task_name,
    _tool_codes,
)


def _route_events(trace: dict[str, Any]) -> list[dict[str, str]]:
    events = []
    for code in _tool_codes(trace):
        experts = SELECTED_EXPERT.findall(code)
        if experts:
            events.extend(
                {"action": "delegate_terminal", "expert_id": expert_id}
                for expert_id in experts
            )
        elif "specialist_manager = await rlm" in code:
            events.append({"action": "delegate_coordinator", "expert_id": "none"})
        elif "specialist_local_values" in code:
            events.append({"action": "solve_owned", "expert_id": "none"})
    return events


def _expected_routes(trace: dict[str, Any]) -> list[dict[str, str]]:
    family = _task_data(trace).get("family")
    if family == "specialist_local":
        return [{"action": "solve_owned", "expert_id": "none"}]
    if family == "specialist_generic":
        expert_id = "generic_worker"
    elif family in {"specialist_table_join", "specialist_table_reconcile"}:
        expert_id = "table_analyst"
    elif family in {"specialist_source_ast", "specialist_source_config"}:
        expert_id = "source_inspector"
    elif family == "specialist_recursive_table":
        return [
            {"action": "delegate_coordinator", "expert_id": "none"},
            {"action": "delegate_terminal", "expert_id": "table_analyst"},
        ]
    elif family == "specialist_recursive_source":
        return [
            {"action": "delegate_coordinator", "expert_id": "none"},
            {"action": "delegate_terminal", "expert_id": "source_inspector"},
        ]
    else:
        raise ValueError(f"unsupported routing-screen family: {family}")
    return [{"action": "delegate_terminal", "expert_id": expert_id}]


def _row(trace: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_routes(trace)
    observed = _route_events(trace)
    return {
        "task": _task_name(trace),
        "family": _task_data(trace).get("family"),
        "expected_routes": expected,
        "observed_routes": observed,
        "root_route_correct": bool(observed) and observed[0] == expected[0],
        "all_routes_correct": observed == expected,
        "local_did_not_delegate": (
            _task_data(trace).get("family") != "specialist_local"
            or (
                observed == [{"action": "solve_owned", "expert_id": "none"}]
                and (trace.get("metrics") or {}).get("coordination_spawn_calls") == 0
            )
        ),
        "recursive_manager_route_correct": (
            None
            if len(expected) == 1
            else len(observed) >= 2 and observed[1] == expected[1]
        ),
    }


def summarize(
    *,
    traces_path: Path,
    audit_path: Path,
    expected_models: dict[str, str],
    expected_tasks: int = 16,
) -> dict[str, Any]:
    traces = _load_traces(traces_path)
    if len(traces) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} traces, found {len(traces)}")
    rows = [_row(trace) for trace in traces]
    if len({_row["task"] for _row in rows}) != expected_tasks:
        raise ValueError("routing-screen task names are not unique")

    routes = _audit_terminal_routes(_load_jsonl(audit_path))
    trace_terminal_experts = [
        event["expert_id"]
        for row in rows
        for event in row["observed_routes"]
        if event["action"] == "delegate_terminal"
    ]
    audit_terminal_experts = [route["expert_id"] for route in routes]
    exact_route_sequence = trace_terminal_experts == audit_terminal_experts
    worker_models_exact = all(
        route["expert_id"] in expected_models
        and route["worker_model_call_count"] > 0
        and route["worker_models"] == [expected_models[route["expert_id"]]]
        for route in routes
    )
    family_root_correct = Counter(
        row["family"] for row in rows if row["root_route_correct"]
    )
    recursive_rows = [
        row for row in rows if row["recursive_manager_route_correct"] is not None
    ]
    source_rows = [
        row for row in rows if row["family"].startswith("specialist_source_")
    ]
    table_rows = [row for row in rows if row["family"].startswith("specialist_table_")]
    route_classes = Counter(
        f"{event['action']}:{event['expert_id']}"
        for row in rows
        for event in row["observed_routes"]
    )
    root_correct = sum(row["root_route_correct"] for row in rows)
    route_class_presence = {
        f"{expected[0]['action']}:{expected[0]['expert_id']}"
        for expected in (_expected_routes(trace) for trace in traces)
    }
    observed_correct_classes = {
        f"{row['expected_routes'][0]['action']}:{row['expected_routes'][0]['expert_id']}"
        for row in rows
        if row["root_route_correct"]
    }
    acceptance = {
        "minimum_correct_root_routes": root_correct >= 12,
        "minimum_correct_per_root_route_class": route_class_presence
        <= observed_correct_classes,
        "minimum_correct_source_routes": sum(
            row["root_route_correct"] for row in source_rows
        )
        >= 3,
        "minimum_correct_table_routes": sum(
            row["root_route_correct"] for row in table_rows
        )
        >= 3,
        "minimum_correct_recursive_root_routes": sum(
            row["root_route_correct"] for row in recursive_rows
        )
        >= 3,
        "minimum_correct_recursive_manager_expert_routes": sum(
            row["recursive_manager_route_correct"] is True for row in recursive_rows
        )
        >= 3,
        "local_tasks_did_not_delegate": all(
            row["local_did_not_delegate"] for row in rows
        ),
        "exact_typed_route_provenance": exact_route_sequence,
        "activated_worker_models_exact": worker_models_exact,
    }
    return {
        "schema_version": "q35-2b-specialist-routing-screen-summary/v1",
        "trace_count": len(rows),
        "acceptance_gates_relaxed": False,
        "root_routes_correct": root_correct,
        "all_routes_correct": sum(row["all_routes_correct"] for row in rows),
        "root_correct_by_family": dict(sorted(family_root_correct.items())),
        "recursive_manager_routes_correct": sum(
            row["recursive_manager_route_correct"] is True for row in recursive_rows
        ),
        "observed_route_classes": dict(sorted(route_classes.items())),
        "exact_route_sequence": exact_route_sequence,
        "activated_worker_models_exact": worker_models_exact,
        "routes": routes,
        "tasks": rows,
        "acceptance": acceptance,
        "routing_screen_gate_passed": all(acceptance.values()),
    }


def _model_map(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        expert_id, separator, model = value.partition("=")
        if not separator or not expert_id or not model or expert_id in result:
            raise ValueError(f"invalid expert model mapping: {value!r}")
        result[expert_id] = model
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-model", action="append", default=[])
    parser.add_argument("--expected-tasks", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        traces_path=args.traces,
        audit_path=args.audit,
        expected_models=_model_map(args.expected_model),
        expected_tasks=args.expected_tasks,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
