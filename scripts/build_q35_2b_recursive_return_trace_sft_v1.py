#!/usr/bin/env python3
"""Build recursive-coordinator SFT from admitted forced-return traces."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "q35-2b-recursive-coordinator-return-trace-sft/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_value(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("score", 0.0)
    return float(value or 0.0)


def branch_paths(trace: dict[str, Any]) -> list[list[int]]:
    nodes = trace["nodes"]
    parents = [node.get("parent") for node in nodes]
    parent_nodes = {parent for parent in parents if parent is not None}
    paths: list[list[int]] = []
    for leaf in (index for index in range(len(nodes)) if index not in parent_nodes):
        path: list[int] = []
        seen: set[int] = set()
        current: int | None = leaf
        while current is not None:
            if current in seen:
                raise ValueError(f"trace {trace.get('id')} has a cycle")
            seen.add(current)
            path.append(current)
            current = parents[current]
        paths.append(list(reversed(path)))
    return paths


def wire_message(message: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in message.items() if value is not None}
    content = result.get("content")
    if isinstance(content, list):
        if not all(isinstance(part, dict) and part.get("type") == "text" for part in content):
            raise ValueError("recursive-return trace SFT supports text-only messages")
        result["content"] = "".join(str(part.get("text", "")) for part in content)
    return result


def _successful_trace(
    episode: dict[str, Any], *, require_hard_success: bool = True
) -> dict[str, Any]:
    traces = episode.get("traces")
    if not isinstance(traces, list) or len(traces) != 1:
        raise ValueError(f"episode {episode.get('id')} must contain exactly one trace")
    trace = traces[0]
    if (
        require_hard_success
        and score_value((trace.get("rewards") or {}).get("harness_score")) != 1.0
    ):
        raise ValueError(f"trace {trace.get('id')} is not a hard success")
    if float((trace.get("metrics") or {}).get("child_action_completed", 0.0)) != 1.0:
        raise ValueError(f"trace {trace.get('id')} lacks a complete delegated return")
    if trace.get("stop_condition") != "user_closed":
        raise ValueError(f"trace {trace.get('id')} did not close cleanly")
    return trace


def is_qualifying_episode(
    episode: dict[str, Any], *, require_hard_success: bool = True
) -> bool:
    traces = episode.get("traces")
    if not isinstance(traces, list) or len(traces) != 1:
        return False
    trace = traces[0]
    return (
        (
            not require_hard_success
            or score_value((trace.get("rewards") or {}).get("harness_score")) == 1.0
        )
        and float((trace.get("metrics") or {}).get("child_action_completed", 0.0))
        == 1.0
        and trace.get("stop_condition") == "user_closed"
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _natural_parent_send(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    awaited_sends = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if _dotted_name(call.func) == "agent_message.send":
            awaited_sends.append(call)
    if len(awaited_sends) != 1:
        return False
    send = awaited_sends[0]
    keywords = {keyword.arg: keyword.value for keyword in send.keywords if keyword.arg}
    receiver = keywords.get("receiver_role")
    receiver_name = keywords.get("receiver_name")
    if receiver_name is not None:
        return False
    if receiver is not None and not (
        isinstance(receiver, ast.Constant) and receiver.value == "parent"
    ):
        return False
    return not any(
        isinstance(node, ast.Call) and _dotted_name(node.func) == "rlm"
        for node in ast.walk(tree)
    )


def recursive_return_row(
    episode: dict[str, Any],
    *,
    allow_natural_action: bool = False,
    require_hard_success: bool = True,
) -> dict[str, Any]:
    trace = _successful_trace(episode, require_hard_success=require_hard_success)
    nodes = trace["nodes"]
    roots = [index for index, node in enumerate(nodes) if node.get("parent") is None]
    if len(roots) != 2:
        raise ValueError(f"trace {trace.get('id')} must contain one delegated session")
    delegated_root = roots[1]
    paths = [path for path in branch_paths(trace) if path[0] == delegated_root]
    if len(paths) != 1:
        raise ValueError(f"trace {trace.get('id')} has {len(paths)} delegated branches")
    path = paths[0]
    sampled = [index for index in path if nodes[index].get("sampled") is True]
    if not sampled:
        raise ValueError(f"trace {trace.get('id')} has no sampled delegated action")
    action_index = sampled[0]
    action = nodes[action_index]["message"]
    tool_calls = action.get("tool_calls") or []
    if len(tool_calls) != 1 or tool_calls[0].get("name") != "ipython":
        raise ValueError(f"trace {trace.get('id')} lacks one IPython return action")
    arguments = json.loads(tool_calls[0]["arguments"])
    child = trace["task"]["data"]["oracle"]["children"][0]
    expected_code = f"await agent_message.send({str(child['expected_result'])!r}, receiver_role='parent')"
    code = arguments.get("code") if isinstance(arguments, dict) else None
    if allow_natural_action:
        if not isinstance(code, str) or not _natural_parent_send(code):
            raise ValueError(f"trace {trace.get('id')} lacks one natural parent send")
    elif arguments != {"code": expected_code}:
        raise ValueError(f"trace {trace.get('id')} return action is not exact")
    action_position = path.index(action_index)
    if action_position + 1 >= len(path):
        raise ValueError(f"trace {trace.get('id')} lacks a tool acknowledgement")
    tool_index = path[action_position + 1]
    tool_message = nodes[tool_index]["message"]
    if tool_message.get("role") != "tool":
        raise ValueError(f"trace {trace.get('id')} return action lacks tool acknowledgement")
    trained_path = path[: action_position + 2]
    family = trace["task"]["data"]["generation_metadata"]["resource_families"][0]
    return {
        "messages": [wire_message(nodes[index]["message"]) for index in trained_path],
        "tools": json.dumps(trace.get("tools") or [], sort_keys=True, separators=(",", ":")),
        "axis": (
            "recursive_coordinator_return_natural_action"
            if allow_natural_action
            else "recursive_coordinator_return_forced_action"
        ),
        "phase": "e0c4_recursive_coordinator_return",
        "task_key": trace["task"]["key"],
        "trace_id": trace["id"],
        "role": "coordinator_nonroot",
        "objective": "exactly_one_parent_send_then_stop",
        "expected_result": str(child["expected_result"]),
        "resource_family": family,
    }


def root_anchor_rows(trace_path: Path, repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            trace = episode["traces"][0]
            if score_value((trace.get("rewards") or {}).get("harness_score")) != 1.0:
                continue
            roots = [index for index, node in enumerate(trace["nodes"]) if node.get("parent") is None]
            primary_root = roots[0]
            paths = [path for path in branch_paths(trace) if path[0] == primary_root]
            path = max(
                paths,
                key=lambda item: (
                    sum(trace["nodes"][index].get("sampled") is True for index in item),
                    len(item),
                ),
            )
            base = {
                "messages": [wire_message(trace["nodes"][index]["message"]) for index in path],
                "tools": json.dumps(trace.get("tools") or [], sort_keys=True, separators=(",", ":")),
                "axis": "root_coordinator_retention",
                "phase": "e0d3_uncapped_yield_exact_child",
                "task_key": trace["task"]["key"],
                "trace_id": trace["id"],
                "role": "coordinator_root",
                "objective": "retain_admitted_root_coordinator_behavior",
                "expected_result": None,
                "resource_family": None,
            }
            rows.extend(dict(base) for _ in range(repeats))
    if not rows:
        raise ValueError(f"no hard-success root anchors in {trace_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forced-return-traces", type=Path, required=True)
    parser.add_argument("--root-anchor-traces", type=Path)
    parser.add_argument("--child-only", action="store_true")
    parser.add_argument("--natural-child-actions", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--return-repeats", type=int, default=4)
    parser.add_argument("--root-anchor-repeats", type=int, default=2)
    parser.add_argument("--minimum-return-traces", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite corpus: {args.output_dir}")
    if args.child_only and args.root_anchor_traces is not None:
        raise ValueError("child-only corpus must not include root anchors")
    if args.natural_child_actions and not args.child_only:
        raise ValueError("natural child actions require --child-only")
    if not args.child_only and args.root_anchor_traces is None:
        raise ValueError("mixed corpus requires --root-anchor-traces")
    if args.return_repeats < 1 or args.root_anchor_repeats < 1 or args.minimum_return_traces < 1:
        raise ValueError("corpus repeat counts must be positive")

    from datasets import Dataset

    returns: list[dict[str, Any]] = []
    source_episodes = 0
    accepted_trajectories = 0
    rejected_task_keys: list[str] = []
    with args.forced_return_traces.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                source_episodes += 1
                episode = json.loads(line)
                if not is_qualifying_episode(
                    episode, require_hard_success=not args.natural_child_actions
                ):
                    traces = episode.get("traces") or []
                    rejected_task_keys.append(
                        traces[0].get("task", {}).get("key", episode.get("id", "unknown"))
                        if traces
                        else episode.get("id", "unknown")
                    )
                    continue
                row = recursive_return_row(
                    episode,
                    allow_natural_action=args.natural_child_actions,
                    require_hard_success=not args.natural_child_actions,
                )
                accepted_trajectories += 1
                returns.extend(dict(row) for _ in range(args.return_repeats))
    if accepted_trajectories < args.minimum_return_traces:
        raise ValueError(
            f"only {accepted_trajectories} qualifying return traces; requires {args.minimum_return_traces}"
        )
    anchors = (
        []
        if args.child_only
        else root_anchor_rows(args.root_anchor_traces, args.root_anchor_repeats)
    )
    rows = [*returns, *anchors]
    args.output_dir.mkdir(parents=True)
    parquet_path = args.output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet_path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "objective": (
            "natural_recursive_child_return_consolidation"
            if args.natural_child_actions
            else "forced_recursive_child_return_consolidation"
            if args.child_only
            else "forced_recursive_coordinator_return_with_root_retention"
        ),
        "row_count": len(rows),
        "recursive_return_rows": len(returns),
        "source_episodes": source_episodes,
        "accepted_return_trajectories": accepted_trajectories,
        "rejected_return_trajectories": source_episodes - accepted_trajectories,
        "rejected_task_keys": rejected_task_keys,
        "minimum_return_traces": args.minimum_return_traces,
        "root_retention_rows": len(anchors),
        "return_repeats": args.return_repeats,
        "root_anchor_repeats": args.root_anchor_repeats,
        "natural_child_actions": args.natural_child_actions,
        "resource_family_counts": dict(sorted(Counter(row["resource_family"] for row in returns).items())),
        "sources": {
            "forced_return_traces": {
                "path": str(args.forced_return_traces),
                "sha256": sha256_file(args.forced_return_traces),
            },
        },
        "dataset": {"path": parquet_path.name, "sha256": sha256_file(parquet_path)},
    }
    if args.root_anchor_traces is not None:
        manifest["sources"]["root_anchor_traces"] = {
            "path": str(args.root_anchor_traces),
            "sha256": sha256_file(args.root_anchor_traces),
        }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows: {len(returns)} recursive return + {len(anchors)} root retention")


if __name__ == "__main__":
    main()
