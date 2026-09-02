#!/usr/bin/env python3
"""Audit a same-task generalist/specialist population comparison."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SELECTED_EXPERT = re.compile(r"\[selected terminal capability\]\s*expert_id=([a-z_]+)")
SPECIALISTS = ("table_analyst", "source_inspector")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _load_traces(path: Path) -> list[dict[str, Any]]:
    traces = []
    for envelope in _load_jsonl(path):
        values = envelope.get("traces") or []
        if not isinstance(values, list):
            raise ValueError(f"{path}: traces must be a list")
        traces.extend(value for value in values if isinstance(value, dict))
    return traces


def _task_data(trace: dict[str, Any]) -> dict[str, Any]:
    value = trace.get("task", {}).get("data")
    if not isinstance(value, dict):
        raise ValueError("trace lacks task data")
    return value


def _task_name(trace: dict[str, Any]) -> str:
    value = _task_data(trace).get("name")
    if not isinstance(value, str) or not value:
        raise ValueError("trace lacks task name")
    return value


def _tool_codes(trace: dict[str, Any]) -> list[str]:
    codes = []
    for node in trace.get("nodes") or []:
        if node.get("sampled") is not True:
            continue
        message = node.get("message") or {}
        for call in message.get("tool_calls") or []:
            arguments = call.get("arguments")
            if arguments is None and isinstance(call.get("function"), dict):
                arguments = call["function"].get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if isinstance(arguments, dict) and isinstance(arguments.get("code"), str):
                codes.append(arguments["code"])
    return codes


def _selected_experts(trace: dict[str, Any]) -> list[str]:
    selected = []
    for code in _tool_codes(trace):
        selected.extend(SELECTED_EXPERT.findall(code))
    return selected


def _model_authored_computation(trace: dict[str, Any]) -> bool:
    return any(
        "agent_message.send" in code and ("read_text" in code or "read_csv" in code or "open(" in code)
        for code in _tool_codes(trace)
    )


def _hard_success(trace: dict[str, Any]) -> bool:
    metrics = trace.get("metrics") or {}
    family = _task_data(trace).get("family")
    return (
        metrics.get("answer_accuracy") == 1
        and metrics.get("protocol_aligned") == 1
        and metrics.get("clean_protocol_aligned") == 1
        and trace.get("stop_condition") == "agent_completed"
        and (family == "specialist_local" or _model_authored_computation(trace))
    )


def _trace_row(trace: dict[str, Any]) -> dict[str, Any]:
    task = _task_data(trace)
    metrics = trace.get("metrics") or {}
    selected = _selected_experts(trace)
    return {
        "task": _task_name(trace),
        "trace_id": trace.get("id"),
        "family": task.get("family"),
        "template_variant": task.get("template_variant"),
        "selected_experts": selected,
        "preferred_expert": task.get("preferred_expert"),
        "answer_accuracy": metrics.get("answer_accuracy"),
        "protocol_aligned": metrics.get("protocol_aligned"),
        "clean_protocol_aligned": metrics.get("clean_protocol_aligned"),
        "spawn_calls": metrics.get("coordination_spawn_calls"),
        "failed_cells": metrics.get("failed_cells"),
        "model_authored_worker_computation": (
            task.get("family") == "specialist_local" or _model_authored_computation(trace)
        ),
        "hard_success": _hard_success(trace),
    }


def _audit_terminal_routes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join coordinator route decisions to the worker calls they activated.

    The typed decision row is served by the coordinator checkpoint.  Subsequent
    ``forwarded`` rows with the same session and expert id are the calls served by
    the selected worker checkpoint.  Keeping the two identities separate avoids
    accidentally treating coordinator provenance as worker provenance.
    """
    worker_models: dict[tuple[Any, Any], list[Any]] = {}
    for row in rows:
        expert_id = row.get("expert_id")
        if row.get("mode") != "forwarded" or not isinstance(expert_id, str):
            continue
        key = (row.get("session_sha256"), expert_id)
        worker_models.setdefault(key, []).append(row.get("upstream_model"))

    routes = []
    for row in rows:
        if "specialist_cognitive_action_delegate_terminal_" not in str(row.get("mode")):
            continue
        expert_id = row.get("expert_id")
        session_sha256 = row.get("session_sha256")
        models = worker_models.get((session_sha256, expert_id), [])
        routes.append(
            {
                "sequence": row.get("sequence"),
                "expert_id": expert_id,
                "coordinator_model": row.get("upstream_model"),
                "worker_models": sorted(set(models), key=str),
                "worker_model_call_count": len(models),
                "latency_ms": row.get("latency_ms"),
                "session_sha256": session_sha256,
            }
        )
    return routes


def _arm(
    traces_path: Path,
    audit_path: Path,
    expected_models: dict[str, str],
    expected_tasks: int,
) -> dict[str, Any]:
    traces = _load_traces(traces_path)
    if len(traces) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} traces in {traces_path}, found {len(traces)}")
    rows = [_trace_row(trace) for trace in traces]
    if len({row["task"] for row in rows}) != expected_tasks:
        raise ValueError(f"{traces_path}: task names are not unique")
    routes = _audit_terminal_routes(_load_jsonl(audit_path))
    trace_experts = [expert for row in rows for expert in row["selected_experts"]]
    audit_experts = [route["expert_id"] for route in routes]
    exact_route_sequence = trace_experts == audit_experts
    route_models_exact = all(
        isinstance(route["expert_id"], str)
        and route["expert_id"] in expected_models
        and route["worker_model_call_count"] > 0
        and route["worker_models"] == [expected_models[route["expert_id"]]]
        for route in routes
    )
    activations = Counter(trace_experts)
    local_rows = [row for row in rows if row["family"] == "specialist_local"]
    return {
        "trace_count": len(rows),
        "hard_successes": sum(row["hard_success"] for row in rows),
        "family_hard_successes": dict(sorted(Counter(row["family"] for row in rows if row["hard_success"]).items())),
        "worker_activations": dict(sorted(activations.items())),
        "exact_route_sequence": exact_route_sequence,
        "route_models_exact": route_models_exact,
        "local_tasks_did_not_delegate": bool(local_rows)
        and all(row["spawn_calls"] == 0 and not row["selected_experts"] for row in local_rows),
        "routes": routes,
        "tasks": rows,
    }


def _pair_contract(trace: dict[str, Any]) -> dict[str, Any]:
    task = _task_data(trace)
    return {
        "name": task.get("name"),
        "family": task.get("family"),
        "template_variant": task.get("template_variant"),
        "answer": task.get("answer"),
        "files": task.get("files"),
        "child_paths": task.get("child_paths"),
    }


def summarize(
    *,
    control_traces: Path,
    control_audit: Path,
    treatment_traces: Path,
    treatment_audit: Path,
    control_models: dict[str, str],
    treatment_models: dict[str, str],
    expected_tasks: int = 16,
    recovery_floor: int = 4,
) -> dict[str, Any]:
    control_trace_values = _load_traces(control_traces)
    treatment_trace_values = _load_traces(treatment_traces)
    control_contracts = {_task_name(trace): _pair_contract(trace) for trace in control_trace_values}
    treatment_contracts = {_task_name(trace): _pair_contract(trace) for trace in treatment_trace_values}
    if control_contracts != treatment_contracts:
        raise ValueError("control and treatment task contracts differ")
    control = _arm(control_traces, control_audit, control_models, expected_tasks)
    treatment = _arm(treatment_traces, treatment_audit, treatment_models, expected_tasks)
    control_rows = {row["task"]: row for row in control["tasks"]}
    treatment_rows = {row["task"]: row for row in treatment["tasks"]}
    paired_recoveries = [
        name
        for name in sorted(control_rows)
        if not control_rows[name]["hard_success"] and treatment_rows[name]["hard_success"]
    ]
    attributed = Counter()
    for name in paired_recoveries:
        for expert_id in treatment_rows[name]["selected_experts"]:
            if expert_id in SPECIALISTS:
                attributed[expert_id] += 1
    worker_activations = treatment["worker_activations"]
    activation_total = sum(worker_activations.values())
    max_fraction = max(worker_activations.values(), default=0) / activation_total if activation_total else 1.0
    acceptance = {
        "minimum_distinct_paired_recoveries": len(paired_recoveries) >= recovery_floor,
        "minimum_recoveries_attributed_to_each_specialist": all(
            attributed[expert_id] >= 1 for expert_id in SPECIALISTS
        ),
        "treatment_clean_total_exceeds_control_by_four": (treatment["hard_successes"] >= control["hard_successes"] + 4),
        "exact_route_provenance": treatment["exact_route_sequence"] and treatment["route_models_exact"],
        "minimum_two_worker_ids_activated": len(worker_activations) >= 2,
        "no_worker_exceeds_three_quarters": max_fraction <= 0.75,
        "coordinator_local_tasks_did_not_delegate": treatment["local_tasks_did_not_delegate"],
    }
    return {
        "schema_version": "q35-2b-specialist-population-paired-summary/v1",
        "task_contracts_identical": True,
        "expected_tasks": expected_tasks,
        "selector_uses_expected_answer": False,
        "acceptance_gates_relaxed": False,
        "control": control,
        "treatment": treatment,
        "paired_recoveries": paired_recoveries,
        "paired_recovery_count": len(paired_recoveries),
        "recoveries_attributed_to_specialists": dict(sorted(attributed.items())),
        "maximum_worker_activation_fraction": max_fraction,
        "acceptance": acceptance,
        "paired_population_gate_passed": all(acceptance.values()),
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
    parser.add_argument("--control-traces", type=Path, required=True)
    parser.add_argument("--control-audit", type=Path, required=True)
    parser.add_argument("--treatment-traces", type=Path, required=True)
    parser.add_argument("--treatment-audit", type=Path, required=True)
    parser.add_argument("--control-model", action="append", default=[])
    parser.add_argument("--treatment-model", action="append", default=[])
    parser.add_argument("--expected-tasks", type=int, default=16)
    parser.add_argument("--recovery-floor", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        control_traces=args.control_traces,
        control_audit=args.control_audit,
        treatment_traces=args.treatment_traces,
        treatment_audit=args.treatment_audit,
        control_models=_model_map(args.control_model),
        treatment_models=_model_map(args.treatment_model),
        expected_tasks=args.expected_tasks,
        recovery_floor=args.recovery_floor,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
