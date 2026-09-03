#!/usr/bin/env python3
"""Audit public-affordance and cost utility in live specialist trajectories."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from summarize_q35_2b_specialist_population_v1 import (
    _audit_terminal_routes,
    _hard_success,
    _load_jsonl,
    _load_traces,
    _task_data,
    _task_name,
)
from summarize_q35_2b_specialist_routing_screen_v1 import _route_events

REGISTRY_HEADER = "[capability registry]"
ASSIGNMENT_HEADER = "[terminal specialist assignment]"


def _public_registry(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    prompt = _task_data(trace).get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"{_task_name(trace)} lacks its public prompt")
    rows: dict[str, dict[str, Any]] = {}
    in_registry = False
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped == REGISTRY_HEADER:
            in_registry = True
            continue
        if stripped == ASSIGNMENT_HEADER:
            in_registry = False
            continue
        if not in_registry or not stripped.startswith("{"):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        expert_id = row.get("expert_id") if isinstance(row, dict) else None
        affordances = row.get("affordances") if isinstance(row, dict) else None
        cost = row.get("relative_cost") if isinstance(row, dict) else None
        if (
            not isinstance(expert_id, str)
            or not isinstance(affordances, list)
            or not affordances
            or not all(isinstance(value, str) and value for value in affordances)
            or not isinstance(cost, (int, float))
            or isinstance(cost, bool)
            or cost <= 0
        ):
            raise ValueError(f"{_task_name(trace)} has an invalid public registry row")
        canonical = {
            "expert_id": expert_id,
            "affordances": tuple(affordances),
            "relative_cost": float(cost),
        }
        if expert_id in rows and rows[expert_id] != canonical:
            raise ValueError(f"{_task_name(trace)} has conflicting public registry rows")
        rows[expert_id] = canonical
    if not rows:
        raise ValueError(f"{_task_name(trace)} lacks a structured public registry")
    return rows


def _public_assignment(trace: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _task_data(trace).get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"{_task_name(trace)} lacks its public prompt")
    assignments = []
    lines = prompt.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip() != ASSIGNMENT_HEADER:
            continue
        try:
            value = json.loads(lines[index + 1].strip())
        except json.JSONDecodeError as error:
            raise ValueError(f"{_task_name(trace)} has invalid public assignment JSON") from error
        if isinstance(value, dict):
            assignments.append(value)
    if not assignments:
        return None
    if any(value != assignments[0] for value in assignments[1:]):
        raise ValueError(f"{_task_name(trace)} has conflicting public assignments")
    return assignments[0]


def _required_affordance(trace: dict[str, Any]) -> str | None:
    assignment = _public_assignment(trace)
    if assignment is None:
        if _task_data(trace).get("family") == "specialist_local":
            return None
        raise ValueError(f"{_task_name(trace)} lacks a public terminal assignment")
    paths = assignment.get("paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
        raise ValueError(f"{_task_name(trace)} has invalid public assignment paths")
    suffixes = {Path(path).suffix.lower() for path in paths}
    if suffixes & {".py", ".toml", ".env"}:
        return "source_config_inspection"
    if ".csv" in suffixes or len(paths) > 1:
        return "multi_artifact_table"
    if suffixes == {".json"} and len(paths) == 1:
        return "single_json_arithmetic"
    raise ValueError(f"{_task_name(trace)} has no public affordance mapping")


def _expected_root_action(trace: dict[str, Any]) -> str:
    family = _task_data(trace).get("family")
    if family == "specialist_local":
        return "solve_owned"
    if isinstance(family, str) and family.startswith("specialist_recursive_"):
        return "delegate_coordinator"
    if isinstance(family, str) and family.startswith("specialist_"):
        return "delegate_terminal"
    raise ValueError(f"unsupported utility-screen family: {family}")


def _row(trace: dict[str, Any], *, utility_cost_lambda: float) -> dict[str, Any]:
    registry = _public_registry(trace)
    required = _required_affordance(trace)
    observed = _route_events(trace)
    expected_root_action = _expected_root_action(trace)
    root_action_correct = bool(observed) and observed[0]["action"] == expected_root_action
    terminal = [event for event in observed if event["action"] == "delegate_terminal"]
    selected = terminal[0]["expert_id"] if len(terminal) == 1 else None
    valid_experts = (
        [
            expert_id
            for expert_id, public in registry.items()
            if required in public["affordances"]
        ]
        if required is not None
        else []
    )
    selected_valid = (
        required is None
        or (
            selected in registry
            and selected in valid_experts
            and len(terminal) == 1
        )
    )
    minimum_cost = (
        min(registry[expert_id]["relative_cost"] for expert_id in valid_experts)
        if valid_experts
        else None
    )
    selected_cost = (
        registry[selected]["relative_cost"] if selected in registry else None
    )
    nondominated = (
        required is None
        or (
            selected_valid
            and minimum_cost is not None
            and selected_cost is not None
            and math.isclose(selected_cost, minimum_cost, rel_tol=0, abs_tol=1e-9)
        )
    )
    hard_success = _hard_success(trace)
    route_utility = (
        1.0 - utility_cost_lambda * selected_cost
        if hard_success and selected_valid and selected_cost is not None
        else (1.0 if hard_success and required is None else 0.0)
    )
    complete = hard_success and root_action_correct and nondominated
    return {
        "task": _task_name(trace),
        "family": _task_data(trace).get("family"),
        "expected_root_action": expected_root_action,
        "observed_routes": observed,
        "root_action_correct": root_action_correct,
        "required_affordance": required,
        "valid_experts": valid_experts,
        "selected_expert": selected,
        "selected_cost": selected_cost,
        "minimum_valid_cost": minimum_cost,
        "capability_valid": selected_valid,
        "utility_nondominated": nondominated,
        "hard_success": hard_success,
        "route_utility": route_utility,
        "complete_qualifying": complete,
        "public_registry": registry,
    }


def summarize(
    *,
    traces_path: Path,
    audit_path: Path,
    expected_models: dict[str, str],
    expected_router_model: str,
    expected_tasks: int = 16,
    minimum_complete_qualifying: int = 4,
    utility_cost_lambda: float = 0.1,
) -> dict[str, Any]:
    traces = _load_traces(traces_path)
    if len(traces) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} traces, found {len(traces)}")
    rows = [_row(trace, utility_cost_lambda=utility_cost_lambda) for trace in traces]
    if len({row["task"] for row in rows}) != expected_tasks:
        raise ValueError("utility-screen task names are not unique")
    registries = [row["public_registry"] for row in rows]
    public_registry_consistent = all(value == registries[0] for value in registries[1:])

    audit_rows = _load_jsonl(audit_path)
    routes = _audit_terminal_routes(audit_rows)
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
    router_events = [
        event
        for event in audit_rows
        if str(event.get("mode", "")).startswith("forwarded_specialist_expert_router_")
    ]
    router_provenance_exact = (
        [event.get("expert_id") for event in router_events] == trace_terminal_experts
        and all(
            event.get("status") == 200
            and event.get("upstream_model") == expected_router_model
            and isinstance(event.get("response_sha256"), str)
            for event in router_events
        )
    )

    terminal_rows = [row for row in rows if row["required_affordance"] is not None]
    table_rows = [row for row in rows if row["required_affordance"] == "multi_artifact_table"]
    source_rows = [row for row in rows if row["required_affordance"] == "source_config_inspection"]
    generic_rows = [row for row in rows if row["required_affordance"] == "single_json_arithmetic"]
    manager_rows = [
        row for row in rows if str(row["family"]).startswith("specialist_recursive_")
    ]
    selections = Counter(
        row["selected_expert"] for row in terminal_rows if row["selected_expert"] is not None
    )
    maximum_share = max(selections.values(), default=0) / max(sum(selections.values()), 1)
    complete = [row["task"] for row in rows if row["complete_qualifying"]]
    root_actions_correct = sum(row["root_action_correct"] for row in rows)
    capability_valid = sum(row["capability_valid"] for row in terminal_rows)
    nondominated = sum(row["utility_nondominated"] for row in terminal_rows)

    acceptance = {
        "minimum_correct_root_actions": root_actions_correct >= 12,
        "all_root_action_classes_represented": {
            "solve_owned",
            "delegate_terminal",
            "delegate_coordinator",
        }
        <= {
            row["expected_root_action"] for row in rows if row["root_action_correct"]
        },
        "minimum_capability_valid_terminal_routes": capability_valid >= 11,
        "minimum_utility_nondominated_terminal_routes": nondominated >= 11,
        "minimum_generic_cheap_paths": sum(
            row["selected_expert"] == "generic_worker" and row["utility_nondominated"]
            for row in generic_rows
        )
        >= 1,
        "minimum_table_niche_routes": sum(row["utility_nondominated"] for row in table_rows) >= 3,
        "minimum_source_niche_routes": sum(row["utility_nondominated"] for row in source_rows) >= 3,
        "minimum_recursive_manager_cells": sum(
            row["root_action_correct"] and row["utility_nondominated"] for row in manager_rows
        )
        >= 3,
        "maximum_selected_expert_share": maximum_share <= 0.75,
        "public_registry_consistent": public_registry_consistent,
        "exact_typed_route_provenance": exact_route_sequence,
        "activated_worker_models_exact": worker_models_exact,
        "isolated_router_provenance_exact": router_provenance_exact,
        "minimum_complete_qualifying_trajectories": len(complete) >= minimum_complete_qualifying,
    }
    return {
        "schema_version": "q35-2b-specialist-utility-screen-summary/v1",
        "trace_count": len(rows),
        "acceptance_gates_relaxed": False,
        "utility_definition": {
            "verified_success_value": 1.0,
            "cost_lambda": utility_cost_lambda,
            "hard_constraint": "selected expert must expose the required public affordance",
            "equivalent_minimum_cost_routes_all_accepted": True,
        },
        "root_actions_correct": root_actions_correct,
        "capability_valid_terminal_routes": capability_valid,
        "utility_nondominated_terminal_routes": nondominated,
        "selected_expert_counts": dict(sorted(selections.items())),
        "maximum_selected_expert_share": maximum_share,
        "public_registry_consistent": public_registry_consistent,
        "exact_route_sequence": exact_route_sequence,
        "activated_worker_models_exact": worker_models_exact,
        "isolated_router_model": expected_router_model,
        "isolated_router_provenance_exact": router_provenance_exact,
        "complete_qualifying_trajectories": complete,
        "complete_qualifying_count": len(complete),
        "minimum_complete_qualifying": minimum_complete_qualifying,
        "routes": routes,
        "tasks": rows,
        "acceptance": acceptance,
        "utility_screen_gate_passed": all(acceptance.values()),
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
    parser.add_argument("--expected-router-model", required=True)
    parser.add_argument("--expected-tasks", type=int, default=16)
    parser.add_argument("--minimum-complete-qualifying", type=int, default=4)
    parser.add_argument("--utility-cost-lambda", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        traces_path=args.traces,
        audit_path=args.audit,
        expected_models=_model_map(args.expected_model),
        expected_router_model=args.expected_router_model,
        expected_tasks=args.expected_tasks,
        minimum_complete_qualifying=args.minimum_complete_qualifying,
        utility_cost_lambda=args.utility_cost_lambda,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
