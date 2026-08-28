#!/usr/bin/env python3
"""Build a computation-grounded recursive-C return corpus with root-C anchors."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "q35-2b-recursive-coordinator-return-sft/v2"
RECURSIVE_HEADER = "[recursive coordinator session contract]"
SUPPORTED_FAMILIES = (
    "json_sum",
    "csv_total",
    "word_count",
    "md_h2",
    "log_error",
    "python_defs",
    "json_max",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_generator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_recursive_return_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import procedural generator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compute_resource(family: str, content: str, operation: str) -> int:
    if family == "json_sum":
        return sum(json.loads(content))
    if family == "csv_total":
        return sum(int(row["amount"]) for row in csv.DictReader(io.StringIO(content)))
    if family == "word_count":
        match = re.fullmatch(r"count exact '([^']+)' tokens", operation)
        if match is None:
            raise ValueError(f"unrecognized word-count operation: {operation}")
        return content.split().count(match.group(1))
    if family == "md_h2":
        return sum(line.startswith("## ") for line in content.splitlines())
    if family == "log_error":
        return sum(line.startswith("ERROR ") for line in content.splitlines())
    if family == "python_defs":
        return sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.parse(content).body
        )
    if family == "json_max":
        return max(json.loads(content).values())
    raise ValueError(f"unsupported recursive-return resource family: {family}")


def computation_code(family: str, content: str, operation: str) -> str:
    prefix = f"evidence = {content!r}\n"
    if family == "json_sum":
        body = "import json\nresult = sum(json.loads(evidence))"
    elif family == "csv_total":
        body = (
            "import csv, io\n"
            "result = sum(int(row['amount']) for row in "
            "csv.DictReader(io.StringIO(evidence)))"
        )
    elif family == "word_count":
        match = re.fullmatch(r"count exact '([^']+)' tokens", operation)
        if match is None:
            raise ValueError(f"unrecognized word-count operation: {operation}")
        body = f"keyword = {match.group(1)!r}\nresult = evidence.split().count(keyword)"
    elif family == "md_h2":
        body = "result = sum(line.startswith('## ') for line in evidence.splitlines())"
    elif family == "log_error":
        body = "result = sum(line.startswith('ERROR ') for line in evidence.splitlines())"
    elif family == "python_defs":
        body = (
            "import ast\n"
            "tree = ast.parse(evidence)\n"
            "result = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) "
            "for node in tree.body)"
        )
    elif family == "json_max":
        body = "import json\nresult = max(json.loads(evidence).values())"
    else:
        raise ValueError(f"unsupported recursive-return resource family: {family}")
    return (
        f"{prefix}{body}\n"
        "await agent_message.send(str(result), receiver_role='parent')"
    )


def select_balanced_episodes(
    generator: Any,
    *,
    start_index: int,
    examples_per_family: int,
    master_seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    index = start_index
    while any(counts[family] < examples_per_family for family in SUPPORTED_FAMILIES):
        row = generator.generate_curriculum_episode(
            "natural_n1a", "train_gen", index, master_seed, "raw_resource"
        )
        child = row["oracle"]["children"][0]
        resource = row["oracle"]["resource_ownership"][child["resource_path"]]
        family = resource["family"]
        if family in SUPPORTED_FAMILIES and counts[family] < examples_per_family:
            selected.append(row)
            counts[family] += 1
        index += 1
        if index - start_index > 10000:
            raise RuntimeError("could not construct balanced recursive-return corpus")
    return selected


def recursive_return_row(
    episode: dict[str, Any], *, template_system_message: dict[str, Any], tools: str
) -> dict[str, Any]:
    oracle = episode["oracle"]
    child = oracle["children"][0]
    path = child["resource_path"]
    ownership = oracle["resource_ownership"][path]
    family = ownership["family"]
    operation = child["operation"]
    content = oracle["private_resources"][path]
    result = compute_resource(family, content, operation)
    if str(result) != str(child["expected_result"]):
        raise ValueError(f"computed result mismatch for {episode['episode_id']}")

    child_prompt = (
        f"You are {child['name']}, a non-root coordinator. Only this session owns "
        f"{path}. You cannot finalize the user's request and must not create descendants. "
        f"Process the supplied private evidence and {operation}. In your first IPython call "
        "send only the resulting integer as a string to receiver_role='parent'. Send "
        "exactly one report to your parent, then stop; do not return the result only as prose."
    )
    injected = (
        f"[task from parent]\n\n{child_prompt}\n\n{RECURSIVE_HEADER}\n"
        "session_role=coordinator\n"
        "is_root=false\n"
        "can_delegate=false\n"
        "can_finalize_user=false\n"
        "return_contract=exactly_one_parent_report\n"
        "You are the non-root coordinator for this bounded subproblem. Do not call rlm or "
        "create descendants. Compute from the inline evidence, send exactly one result to "
        "your parent with agent_message.send, then stop.\n"
        f"Evidence label: {path}\n"
        f"Required review: {operation}\n"
        "Evidence contents:\n"
        f"{content}"
    )
    call_id = "recursive-return-" + hashlib.sha256(
        episode["episode_id"].encode()
    ).hexdigest()[:16]
    code = computation_code(family, content, operation)
    messages = [
        dict(template_system_message),
        {"role": "user", "content": injected},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": (
                f"I will use Python to {operation}, send the computed result exactly once "
                "to my parent, and stop after the tool acknowledgement."
            ),
            "tool_calls": [
                {
                    "id": call_id,
                    "name": "ipython",
                    "arguments": json.dumps({"code": code}, separators=(",", ":")),
                }
            ],
        },
        {
            "role": "tool",
            "content": "message queued",
            "tool_call_id": call_id,
        },
    ]
    return {
        "messages": messages,
        "tools": tools,
        "axis": "recursive_coordinator_return",
        "phase": "e0c4_recursive_coordinator_return",
        "task_key": episode["episode_id"],
        "trace_id": None,
        "role": "coordinator_nonroot",
        "objective": "compute_then_exactly_one_parent_send_then_stop",
        "expected_result": str(result),
        "resource_family": family,
        "source_trace": None,
    }


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
        if not all(
            isinstance(part, dict) and part.get("type") == "text"
            for part in content
        ):
            raise ValueError("root retention supports text-only Prime Agent messages")
        result["content"] = "".join(str(part.get("text", "")) for part in content)
    return result


def root_anchor_rows(trace_path: Path, repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            trace = episode["traces"][0]
            reward = trace.get("rewards", {}).get("harness_score", {}).get("score")
            if reward != 1.0:
                continue
            roots = [i for i, node in enumerate(trace["nodes"]) if node.get("parent") is None]
            primary_root = roots[0]
            paths = [path for path in branch_paths(trace) if path[0] == primary_root]
            if not paths:
                raise ValueError(f"successful trace {trace['id']} lacks a root branch")
            path = max(
                paths,
                key=lambda item: (
                    sum(trace["nodes"][index].get("sampled") is True for index in item),
                    len(item),
                    item[-1],
                ),
            )
            messages = [wire_message(trace["nodes"][index]["message"]) for index in path]
            if not any(message.get("role") == "assistant" for message in messages):
                raise ValueError(f"successful trace {trace['id']} has no assistant target")
            base = {
                "messages": messages,
                "tools": json.dumps(
                    trace.get("tools") or [], sort_keys=True, separators=(",", ":")
                ),
                "axis": "root_coordinator_retention",
                "phase": "e0d3_uncapped_yield_exact_child",
                "task_key": trace["task"]["key"],
                "trace_id": trace["id"],
                "role": "coordinator_root",
                "objective": "retain_admitted_root_coordinator_behavior",
                "expected_result": None,
                "resource_family": None,
                "source_trace": str(trace_path),
            }
            rows.extend(dict(base) for _ in range(repeats))
    if not rows:
        raise ValueError(f"no hard-success root anchors in {trace_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--template-corpus", type=Path, required=True)
    parser.add_argument("--root-anchor-traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=9420200)
    parser.add_argument("--examples-per-family", type=int, default=4)
    parser.add_argument("--root-anchor-repeats", type=int, default=2)
    parser.add_argument("--master-seed", type=int, default=20260816)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite corpus: {args.output_dir}")

    from datasets import Dataset

    template = Dataset.from_parquet(str(args.template_corpus))[0]
    template_system_message = dict(template["messages"][0])
    tools = template["tools"]
    generator = load_generator(args.generator)
    episodes = select_balanced_episodes(
        generator,
        start_index=args.start_index,
        examples_per_family=args.examples_per_family,
        master_seed=args.master_seed,
    )
    return_rows = [
        recursive_return_row(
            episode, template_system_message=template_system_message, tools=tools
        )
        for episode in episodes
    ]
    anchors = root_anchor_rows(args.root_anchor_traces, args.root_anchor_repeats)
    rows = [*return_rows, *anchors]

    args.output_dir.mkdir(parents=True)
    parquet_path = args.output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet_path))
    family_counts = Counter(row["resource_family"] for row in return_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "objective": "computation_grounded_recursive_return_with_root_retention",
        "row_count": len(rows),
        "recursive_return_rows": len(return_rows),
        "root_retention_rows": len(anchors),
        "resource_family_counts": dict(sorted(family_counts.items())),
        "task_keys": [row["task_key"] for row in return_rows],
        "task_index_range": {
            "requested_start": args.start_index,
            "selected_min": min(episode["index"] for episode in episodes),
            "selected_max": max(episode["index"] for episode in episodes),
        },
        "leak_policy": {
            "raw_evidence_visible": True,
            "gold_answer_in_user_prompt": False,
            "target_uses_python_computation": True,
            "target_contains_exactly_one_parent_send": True,
            "target_contains_no_post_ack_assistant_turn": True,
        },
        "sources": {
            "generator": {
                "path": str(args.generator),
                "sha256": sha256_file(args.generator),
            },
            "template_corpus": {
                "path": str(args.template_corpus),
                "sha256": sha256_file(args.template_corpus),
            },
            "root_anchor_traces": {
                "path": str(args.root_anchor_traces),
                "sha256": sha256_file(args.root_anchor_traces),
            },
        },
        "dataset": {"path": parquet_path.name, "sha256": sha256_file(parquet_path)},
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(rows)} rows: {len(return_rows)} recursive return + "
        f"{len(anchors)} root retention"
    )


if __name__ == "__main__":
    main()
