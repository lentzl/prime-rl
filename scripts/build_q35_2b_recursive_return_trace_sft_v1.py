#!/usr/bin/env python3
"""Build recursive-coordinator SFT from admitted forced-return traces."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dual_policy_openai_proxy_v1 import (
    LEAF_REPORTER_CONTRACT,
    LEAF_REPORTER_HEADER,
    leaf_compute_report_code,
)

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


def with_leaf_reporter_contract(row: dict[str, Any]) -> dict[str, Any]:
    """Render a child SFT row in the same leading context used by the live proxy."""

    messages = copy.deepcopy(row["messages"])
    if any(
        LEAF_REPORTER_HEADER in str(message.get("content", ""))
        for message in messages
    ):
        raise ValueError("child SFT row already contains a leaf reporter contract")
    return {
        **row,
        "messages": [
            {"role": "system", "content": LEAF_REPORTER_CONTRACT},
            *messages,
        ],
    }


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


def canonical_scaffold_compute_code(code: str, operation: str) -> str:
    """Remove only the runtime evidence binding from a verified scaffold action."""

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError("scaffolded compute-report action is not valid Python") from exc
    if not tree.body or not isinstance(tree.body[0], ast.Assign):
        raise ValueError("scaffolded compute-report action lacks INLINE_EVIDENCE binding")
    assignment = tree.body[0]
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or assignment.targets[0].id != "INLINE_EVIDENCE"
        or not isinstance(assignment.value, ast.Constant)
        or not isinstance(assignment.value.value, str)
    ):
        raise ValueError("scaffolded compute-report action has an invalid evidence binding")
    canonical = leaf_compute_report_code(operation)
    observed = ast.Module(body=tree.body[1:], type_ignores=[])
    if ast.dump(observed, include_attributes=False) != ast.dump(
        ast.parse(canonical), include_attributes=False
    ):
        raise ValueError("scaffolded compute-report action differs from canonical operation")
    return canonical


def recursive_return_row(
    episode: dict[str, Any],
    *,
    allow_natural_action: bool = False,
    scaffolded_compute_action: bool = False,
    require_hard_success: bool = True,
) -> dict[str, Any]:
    if allow_natural_action and scaffolded_compute_action:
        raise ValueError("natural and scaffolded compute actions are mutually exclusive")
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
    canonical_code = None
    if scaffolded_compute_action:
        if not isinstance(code, str) or not _natural_parent_send(code):
            raise ValueError(f"trace {trace.get('id')} lacks one scaffolded parent send")
        canonical_code = canonical_scaffold_compute_code(code, child["operation"])
    elif allow_natural_action:
        if not isinstance(code, str) or not _natural_parent_send(code):
            raise ValueError(f"trace {trace.get('id')} lacks one natural parent send")
    elif arguments != {"code": expected_code}:
        raise ValueError(f"trace {trace.get('id')} return action is not exact")
    action_position = path.index(action_index)
    trained_path = path[: action_position + 1]
    if action_position + 1 >= len(path):
        if not allow_natural_action:
            raise ValueError(f"trace {trace.get('id')} lacks a tool acknowledgement")
    else:
        tool_index = path[action_position + 1]
        tool_message = nodes[tool_index]["message"]
        if tool_message.get("role") != "tool":
            raise ValueError(f"trace {trace.get('id')} return action lacks tool acknowledgement")
        trained_path.append(tool_index)
    family = trace["task"]["data"]["generation_metadata"]["resource_families"][0]
    messages = []
    for index in trained_path:
        message = wire_message(copy.deepcopy(nodes[index]["message"]))
        if index == action_index and canonical_code is not None:
            rewritten_arguments = json.loads(message["tool_calls"][0]["arguments"])
            rewritten_arguments["code"] = canonical_code
            message["tool_calls"][0]["arguments"] = json.dumps(
                rewritten_arguments, separators=(",", ":")
            )
        messages.append(message)
    return {
        "messages": messages,
        "tools": json.dumps(trace.get("tools") or [], sort_keys=True, separators=(",", ":")),
        "axis": (
            "recursive_coordinator_return_scaffolded_compute"
            if scaffolded_compute_action
            else "recursive_coordinator_return_natural_action"
            if allow_natural_action
            else "recursive_coordinator_return_forced_action"
        ),
        "phase": "e0c4_recursive_coordinator_return",
        "task_key": trace["task"]["key"],
        "trace_id": trace["id"],
        "role": "coordinator_nonroot",
        "objective": (
            "compute_from_inline_evidence_then_one_parent_send"
            if scaffolded_compute_action
            else "exactly_one_parent_send_then_stop"
        ),
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


def replay_anchor_rows(corpus: Path, repeats: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = corpus / "MANIFEST.json"
    parquet_path = corpus / "train.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("root_retention_rows") != 0
        or manifest.get("objective")
        not in {
            "forced_recursive_child_return_consolidation",
            "natural_recursive_child_return_consolidation",
        }
    ):
        raise ValueError("replay anchor is not a complete child-only return corpus")
    if not parquet_path.is_file() or sha256_file(parquet_path) != manifest.get(
        "dataset", {}
    ).get("sha256"):
        raise ValueError("replay anchor parquet SHA-256 mismatch")

    from datasets import Dataset

    source_rows = list(Dataset.from_parquet(str(parquet_path)))
    if not source_rows or any(row.get("role") != "coordinator_nonroot" for row in source_rows):
        raise ValueError("replay anchor contains a non-child row")
    rows = [dict(row) for _ in range(repeats) for row in source_rows]
    return rows, {
        "path": str(corpus),
        "manifest_sha256": sha256_file(manifest_path),
        "parquet_sha256": sha256_file(parquet_path),
    }


def interleave_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(max((len(group) for group in groups), default=0)):
        rows.extend(group[index] for group in groups if index < len(group))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forced-return-traces", type=Path, action="append", required=True
    )
    parser.add_argument("--root-anchor-traces", type=Path)
    parser.add_argument("--child-only", action="store_true")
    parser.add_argument("--natural-child-actions", action="store_true")
    parser.add_argument("--scaffolded-compute-actions", action="store_true")
    parser.add_argument("--leaf-reporter-contract", action="store_true")
    parser.add_argument("--replay-anchor-corpus", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--return-repeats", type=int, default=4)
    parser.add_argument("--root-anchor-repeats", type=int, default=2)
    parser.add_argument("--replay-anchor-repeats", type=int, default=1)
    parser.add_argument("--minimum-return-traces", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite corpus: {args.output_dir}")
    if args.child_only and args.root_anchor_traces is not None:
        raise ValueError("child-only corpus must not include root anchors")
    if args.natural_child_actions and not args.child_only:
        raise ValueError("natural child actions require --child-only")
    if args.scaffolded_compute_actions and not args.child_only:
        raise ValueError("scaffolded compute actions require --child-only")
    if args.leaf_reporter_contract and not args.child_only:
        raise ValueError("leaf reporter contract requires --child-only")
    if args.natural_child_actions and args.scaffolded_compute_actions:
        raise ValueError("natural and scaffolded compute actions are mutually exclusive")
    if args.replay_anchor_corpus is not None and not args.child_only:
        raise ValueError("replay anchors require --child-only")
    if not args.child_only and args.root_anchor_traces is None:
        raise ValueError("mixed corpus requires --root-anchor-traces")
    if (
        args.return_repeats < 1
        or args.root_anchor_repeats < 1
        or args.replay_anchor_repeats < 1
        or args.minimum_return_traces < 1
    ):
        raise ValueError("corpus repeat counts must be positive")

    from datasets import Dataset

    accepted_rows: list[dict[str, Any]] = []
    source_episodes = 0
    accepted_trajectories = 0
    rejected_task_keys: list[str] = []
    seen_task_keys: set[str] = set()
    for trace_path in args.forced_return_traces:
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source_episodes += 1
                episode = json.loads(line)
                traces = episode.get("traces") or []
                task_key = (
                    traces[0].get("task", {}).get("key", episode.get("id", "unknown"))
                    if traces
                    else episode.get("id", "unknown")
                )
                if task_key in seen_task_keys:
                    raise ValueError(f"duplicate return task across trace banks: {task_key}")
                seen_task_keys.add(task_key)
                if not is_qualifying_episode(
                    episode, require_hard_success=not args.natural_child_actions
                ):
                    rejected_task_keys.append(task_key)
                    continue
                row = recursive_return_row(
                    episode,
                    allow_natural_action=args.natural_child_actions,
                    scaffolded_compute_action=args.scaffolded_compute_actions,
                    require_hard_success=not args.natural_child_actions,
                )
                accepted_trajectories += 1
                accepted_rows.append(row)
    if accepted_trajectories < args.minimum_return_traces:
        raise ValueError(
            f"only {accepted_trajectories} qualifying return traces; requires {args.minimum_return_traces}"
        )
    returns = [
        dict(row) for _ in range(args.return_repeats) for row in accepted_rows
    ]
    root_anchors = (
        []
        if args.child_only
        else root_anchor_rows(args.root_anchor_traces, args.root_anchor_repeats)
    )
    replay_anchors: list[dict[str, Any]] = []
    replay_source = None
    if args.replay_anchor_corpus is not None:
        replay_anchors, replay_source = replay_anchor_rows(
            args.replay_anchor_corpus, args.replay_anchor_repeats
        )
    if args.leaf_reporter_contract:
        returns = [with_leaf_reporter_contract(row) for row in returns]
        replay_anchors = [with_leaf_reporter_contract(row) for row in replay_anchors]
    rows = interleave_rows(returns, replay_anchors, root_anchors)
    args.output_dir.mkdir(parents=True)
    parquet_path = args.output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet_path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "objective": (
            "scaffolded_compute_report_curriculum"
            if args.scaffolded_compute_actions
            else "natural_recursive_child_return_consolidation"
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
        "root_retention_rows": len(root_anchors),
        "replay_anchor_rows": len(replay_anchors),
        "return_repeats": args.return_repeats,
        "root_anchor_repeats": args.root_anchor_repeats,
        "replay_anchor_repeats": args.replay_anchor_repeats,
        "natural_child_actions": args.natural_child_actions,
        "scaffolded_compute_actions": args.scaffolded_compute_actions,
        "leaf_reporter_contract": args.leaf_reporter_contract,
        "resource_family_counts": dict(sorted(Counter(row["resource_family"] for row in returns).items())),
        "sources": {
            "forced_return_traces": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in args.forced_return_traces
            ],
        },
        "dataset": {"path": parquet_path.name, "sha256": sha256_file(parquet_path)},
    }
    if args.root_anchor_traces is not None:
        manifest["sources"]["root_anchor_traces"] = {
            "path": str(args.root_anchor_traces),
            "sha256": sha256_file(args.root_anchor_traces),
        }
    if replay_source is not None:
        manifest["sources"]["replay_anchor_corpus"] = replay_source
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(rows)} rows: {len(returns)} recursive return + "
        f"{len(replay_anchors)} replay anchor + {len(root_anchors)} root retention"
    )


if __name__ == "__main__":
    main()
