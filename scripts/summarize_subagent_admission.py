#!/usr/bin/env python3
"""Summarize the first-turn admission invariant in single-child traces."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _branch_root(nodes: list[dict[str, Any]], node_index: int) -> int:
    seen: set[int] = set()
    while (parent := nodes[node_index].get("parent")) is not None:
        if node_index in seen:
            break
        seen.add(node_index)
        node_index = parent
    return node_index


def _coordinator_cells(trace: dict[str, Any]) -> list[str]:
    nodes = trace["nodes"]
    coordinator_root = _branch_root(nodes, 0)
    cells: list[str] = []
    for node_index, node in enumerate(nodes):
        if _branch_root(nodes, node_index) != coordinator_root:
            continue
        message = node.get("message", {})
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("name") != "ipython":
                continue
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            code = arguments.get("code")
            if isinstance(code, str):
                cells.append(code)
    return cells


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return f"{call.func.value.id}.{call.func.attr}"
    return None


def _assigned_calls(tree: ast.AST) -> set[int]:
    assigned: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if isinstance(value, ast.Await):
            value = value.value
        if isinstance(value, ast.Call):
            assigned.add(id(value))
    return assigned


def _keyword(call: ast.Call, name: str) -> Any:
    value = next((item.value for item in call.keywords if item.arg == name), None)
    return value.value if isinstance(value, ast.Constant) else None


def _spawn_prompt(call: ast.Call) -> str | None:
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None
    value = call.args[0].value
    return value if isinstance(value, str) else None


def _path_used_outside_spawn(tree: ast.AST, path: str) -> bool:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if path not in node.value:
            continue
        ancestor: ast.AST | None = node
        inside_spawn = False
        while ancestor in parents:
            ancestor = parents[ancestor]
            if isinstance(ancestor, ast.Call) and _call_name(ancestor) == "rlm":
                inside_spawn = True
                break
        if not inside_spawn:
            return True
    return False


def _statement_index(tree: ast.AST, target: ast.AST) -> int | None:
    body = getattr(tree, "body", [])
    for index, statement in enumerate(body):
        if any(node is target for node in ast.walk(statement)):
            return index
    return None


def _local_sum(statement: ast.AST) -> bool:
    return "local" in ast.unparse(statement) and any(
        isinstance(node, ast.Call) and _call_name(node) == "sum"
        for node in ast.walk(statement)
    )


def _local_sum_after(tree: ast.AST, statement_index: int) -> bool:
    body = getattr(tree, "body", [])
    return any(_local_sum(statement) for statement in body[statement_index + 1 :])


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    task = trace["task"]["data"]
    if task.get("family") != "single":
        raise ValueError(f"expected single family, got {task.get('family')!r}")
    path = task["child_paths"]["shard-worker"]
    cells = _coordinator_cells(trace)

    first_tree: ast.AST | None = None
    if cells:
        try:
            first_tree = ast.parse(cells[0])
        except SyntaxError:
            pass

    spawn_calls: list[ast.Call] = []
    retained = False
    exact_payload = False
    parent_read_remote = False
    spawn_precedes_local = False
    same_cell_local_work = False
    if first_tree is not None:
        assigned = _assigned_calls(first_tree)
        spawn_calls = [
            node
            for node in ast.walk(first_tree)
            if isinstance(node, ast.Call) and _call_name(node) == "rlm"
        ]
        retained = len(spawn_calls) == 1 and id(spawn_calls[0]) in assigned
        if len(spawn_calls) == 1:
            prompt = _spawn_prompt(spawn_calls[0]) or ""
            exact_payload = (
                _keyword(spawn_calls[0], "name") == "shard-worker"
                and path in prompt
                and "agent_message" in prompt
                and "parent" in prompt
            )
            spawn_index = _statement_index(first_tree, spawn_calls[0])
            spawn_precedes_local = spawn_index is not None and not any(
                _local_sum(statement)
                for statement in getattr(first_tree, "body", [])[:spawn_index]
            )
            same_cell_local_work = spawn_index is not None and _local_sum_after(
                first_tree, spawn_index
            )
        parent_read_remote = _path_used_outside_spawn(first_tree, path)

    separate_local_work = False
    if len(cells) >= 2:
        try:
            second_tree = ast.parse(cells[1])
        except SyntaxError:
            second_tree = None
        if second_tree is not None:
            second_calls = [
                node for node in ast.walk(second_tree) if isinstance(node, ast.Call)
            ]
            separate_local_work = (
                not any(_call_name(call) == "rlm" for call in second_calls)
                and not _path_used_outside_spawn(second_tree, path)
                and any(_call_name(call) == "sum" for call in second_calls)
                and "local" in cells[1]
            )

    exact_admission = (
        len(spawn_calls) == 1
        and retained
        and exact_payload
        and not parent_read_remote
    )
    local_work_after_spawn = same_cell_local_work or separate_local_work
    return {
        "task": task.get("name"),
        "cells": len(cells),
        "spawn_first_cell": len(spawn_calls) == 1,
        "retained_handle": retained,
        "exact_payload": exact_payload,
        "parent_read_remote_first_cell": parent_read_remote,
        "exact_admission": exact_admission,
        "spawn_precedes_local": spawn_precedes_local,
        "separate_local_work": separate_local_work,
        "local_work_after_spawn": local_work_after_spawn,
        "two_stage_admission": exact_admission and local_work_after_spawn,
        "complete_admission": exact_admission
        and spawn_precedes_local
        and local_work_after_spawn,
    }


def summarize_file(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped_traces = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        traces = record.get("traces") or [record]
        for trace in traces:
            if trace.get("task", {}).get("data", {}).get("family") != "single":
                continue
            if not trace.get("nodes"):
                skipped_traces += 1
                continue
            rows.append(summarize_trace(trace))

    fields = (
        "spawn_first_cell",
        "retained_handle",
        "exact_payload",
        "parent_read_remote_first_cell",
        "exact_admission",
        "spawn_precedes_local",
        "separate_local_work",
        "local_work_after_spawn",
        "two_stage_admission",
        "complete_admission",
    )
    return {
        "traces": len(rows),
        "skipped_traces": skipped_traces,
        "counts": {field: sum(bool(row[field]) for row in rows) for field in fields},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--rows", action="store_true", help="include per-trace results")
    args = parser.parse_args()
    summary = summarize_file(args.traces)
    if not args.rows:
        summary.pop("rows")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
