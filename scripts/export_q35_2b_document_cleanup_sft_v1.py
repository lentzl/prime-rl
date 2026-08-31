#!/usr/bin/env python3
"""Export role-separated cleanup replay from admitted document trajectories."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import STEMS, sha256_file

SCHEMA_VERSIONS = {
    "coordinator": "qwen35-2b-document-coordinator-cleanup-sft/v1",
    "child": "qwen35-2b-document-child-cleanup-sft/v1",
}
OBJECTIVES = {
    "coordinator": "successful_on_policy_document_coordinator_cleanup",
    "child": "successful_on_policy_document_child_cleanup",
}
REQUIRED_METRICS = {
    "answer_accuracy": 1,
    "protocol_aligned": 1,
    "spawn_calls": 3,
    "failed_spawn_calls": 0,
    "retained_handles": 3,
    "named_children": 3,
    "delegated_payloads": 3,
    "coordinator_delegated_path_accesses": 0,
    "messages_to_parent": 3,
    "fan_in_complete": 1,
}


def _text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if not all(
            isinstance(part, dict) and part.get("type") == "text" for part in content
        ):
            raise ValueError("document cleanup supports text-only message parts")
        return "".join(str(part.get("text", "")) for part in content)
    raise ValueError("document cleanup message content is not textual")


def _wire_message(message: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy({key: value for key, value in message.items() if value is not None})
    result["content"] = _text(result)
    calls = []
    for call in result.get("tool_calls") or []:
        if "function" in call:
            calls.append(call)
            continue
        if not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), str):
            raise ValueError("document cleanup found an invalid trace tool call")
        calls.append(
            {
                "id": call.get("id"),
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
        )
    if calls or "tool_calls" in result:
        result["tool_calls"] = calls
    return result


def _path_to_node(nodes: list[dict[str, Any]], target: int) -> list[int]:
    path = []
    seen = set()
    current: int | None = target
    while current is not None:
        if current in seen or not 0 <= current < len(nodes):
            raise ValueError("document cleanup trace has an invalid node lineage")
        seen.add(current)
        path.append(current)
        parent = nodes[current].get("parent")
        if parent is not None and not isinstance(parent, int):
            raise ValueError("document cleanup trace has a non-integer parent")
        current = parent
    return list(reversed(path))


def _tool_code(message: dict[str, Any]) -> str | None:
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return None
    call = calls[0]
    if call.get("name") != "ipython":
        return None
    arguments = json.loads(call.get("arguments", "{}"))
    code = arguments.get("code")
    return code if isinstance(code, str) else None


def _admitted_trace(trace: dict[str, Any]) -> None:
    task = trace.get("task", {}).get("data", {})
    metrics = trace.get("metrics") or {}
    if task.get("family") != "document_flat" or not isinstance(task.get("name"), str):
        raise ValueError("cleanup source is not a stable flat document task")
    mismatches = {
        key: (metrics.get(key), value)
        for key, value in REQUIRED_METRICS.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(f"cleanup source is not an admitted trajectory: {mismatches}")
    if metrics.get("clean_protocol_aligned") != 0 or not metrics.get("failed_cells"):
        raise ValueError("cleanup source does not expose the admitted cleanliness defect")


def _root_prefix(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    nodes = trace["nodes"]
    if len(nodes) < 4 or nodes[0].get("parent") is not None:
        raise ValueError("cleanup source lacks its root coordinator prefix")
    task_candidates = [
        index
        for index, node in enumerate(nodes)
        if node.get("parent") == 0
        and node.get("sampled") is False
        and node.get("message", {}).get("role") == "user"
    ]
    if len(task_candidates) != 1:
        raise ValueError("cleanup source lacks one root task prompt")
    task_index = task_candidates[0]
    spawn_candidates = []
    for index, node in enumerate(nodes):
        if node.get("parent") != task_index or node.get("sampled") is not True:
            continue
        code = _tool_code(node.get("message") or {})
        if code is not None and code.count("await rlm(") == 3:
            spawn_candidates.append(index)
    if len(spawn_candidates) != 1:
        raise ValueError("cleanup source lacks one successful three-child spawn")
    spawn_index = spawn_candidates[0]
    receipt_candidates = [
        index
        for index, node in enumerate(nodes)
        if node.get("parent") == spawn_index
        and node.get("sampled") is False
        and node.get("message", {}).get("role") == "tool"
        and "Traceback" not in _text(node["message"])
    ]
    if len(receipt_candidates) != 1:
        raise ValueError("cleanup source lacks one successful spawn receipt")
    return (
        [_wire_message(nodes[index]["message"]) for index in (0, task_index, spawn_index, receipt_candidates[0])],
        spawn_index,
        receipt_candidates[0],
    )


def _passive_targets(traces: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    targets = []
    for trace in traces:
        nodes = trace["nodes"]
        _, _, receipt_index = _root_prefix(trace)
        children = [
            node
            for node in nodes
            if node.get("parent") == receipt_index and node.get("sampled") is True
        ]
        if len(children) != 1:
            continue
        message = children[0].get("message") or {}
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            targets.append((trace["id"], _wire_message(message)))
    if not targets:
        raise ValueError("cleanup sources contain no successful passive-yield target")
    return targets


def _report_messages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    reports = []
    seen = set()
    for node in trace["nodes"]:
        message = node.get("message") or {}
        text = _text(message)
        if message.get("role") != "user" or not text.startswith("[from child:"):
            continue
        matches = [stem for stem in STEMS if f"[from child:{stem}-document-worker]" in text]
        if len(matches) != 1 or matches[0] in seen:
            raise ValueError("cleanup source has ambiguous child report provenance")
        payload = json.loads(text.rsplit("\n\n", 1)[-1])
        if set(payload) != {"words", "h2"} or not all(
            isinstance(value, int) for value in payload.values()
        ):
            raise ValueError("cleanup source has an invalid child report payload")
        seen.add(matches[0])
        reports.append(_wire_message(message))
    if seen != set(STEMS):
        raise ValueError("cleanup source does not contain all three explicit reports")
    return reports


def _report_stem(message: dict[str, Any]) -> str:
    text = _text(message)
    matches = [stem for stem in STEMS if f"[from child:{stem}-document-worker]" in text]
    if len(matches) != 1:
        raise ValueError("cleanup report does not identify one document worker")
    return matches[0]


def _final_message(trace: dict[str, Any]) -> dict[str, Any]:
    answer = trace.get("task", {}).get("data", {}).get("answer")
    candidates = []
    for node in trace["nodes"]:
        if node.get("sampled") is not True:
            continue
        message = node.get("message") or {}
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        try:
            value = json.loads(_text(message).strip())
        except json.JSONDecodeError:
            continue
        if value == answer:
            candidates.append(message)
    if len(candidates) != 1:
        raise ValueError("cleanup source lacks one exact final coordinator answer")
    return _wire_message(candidates[0])


def _partial_target(remaining: list[str]) -> dict[str, Any]:
    names = ", ".join(f"{stem}-document-worker" for stem in remaining)
    return {
        "role": "assistant",
        "content": f"Stored the explicit report; waiting for {names}.",
        "reasoning_content": (
            "This is a partial fan-in. I must preserve the delivered evidence, avoid polling "
            "or inspecting child handles, and yield until every required report arrives."
        ),
        "tool_calls": [],
    }


def _coordinator_rows(traces: list[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    passive_targets = _passive_targets(traces)
    rows = []
    for index, trace in enumerate(traces):
        prefix, _, _ = _root_prefix(trace)
        passive_source, passive = passive_targets[index % len(passive_targets)]
        reports = _report_messages(trace)
        report_stems = [_report_stem(report) for report in reports]
        after_first = [stem for stem in STEMS if stem not in report_stems[:1]]
        after_second = [stem for stem in STEMS if stem not in report_stems[:2]]
        final = _final_message(trace)
        task_key = trace["task"]["data"]["name"]
        rows.extend(
            [
                {
                    "messages": [*copy.deepcopy(prefix), copy.deepcopy(passive)],
                    "family": "document_cleanup_coordinator_passive_yield",
                    "phase": "passive_yield",
                },
                {
                    "messages": [
                        *copy.deepcopy(prefix),
                        copy.deepcopy(passive),
                        copy.deepcopy(reports[0]),
                        _partial_target(after_first),
                    ],
                    "family": "document_cleanup_coordinator_partial_fanin",
                    "phase": "partial_fanin",
                },
                {
                    "messages": [
                        *copy.deepcopy(prefix),
                        copy.deepcopy(passive),
                        copy.deepcopy(reports[0]),
                        _partial_target(after_first),
                        copy.deepcopy(reports[1]),
                        _partial_target(after_second),
                        copy.deepcopy(reports[2]),
                        final,
                    ],
                    "family": "document_cleanup_coordinator_complete_fanin",
                    "phase": "complete_fanin",
                },
            ]
        )
        for row in rows[-3:]:
            row.update(
                {
                    "tools": json.dumps(trace.get("tools") or [], sort_keys=True, separators=(",", ":")),
                    "task_key": f"{task_key}:{row['phase']}",
                    "trace_id": f"document-cleanup:coordinator:{trace['id']}:{row['phase']}",
                    "source_trace_id": trace["id"],
                    "passive_target_source_trace_id": passive_source,
                    "role": "coordinator",
                    "objective": OBJECTIVES["coordinator"],
                    "source_trace": str(source),
                }
            )
    return rows


def _child_rows(trace: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    nodes = trace["nodes"]
    rows = []
    for root_index, root in enumerate(nodes):
        if root.get("parent") is not None:
            continue
        runtime = root.get("message") or {}
        if "Recursive agent depth: 1" not in _text(runtime) or "child agent" not in _text(runtime):
            continue
        terminal_candidates = [
            index
            for index, node in enumerate(nodes)
            if node.get("sampled") is True
            and node.get("message", {}).get("role") == "assistant"
            and _text(node.get("message") or {}).strip() == "Done."
            and root_index in _path_to_node(nodes, index)
        ]
        if len(terminal_candidates) != 1:
            raise ValueError("cleanup child branch lacks one successful terminal response")
        path = _path_to_node(nodes, terminal_candidates[0])
        if len(path) < 5 or path[0] != root_index:
            raise ValueError("cleanup child branch has an invalid successful lineage")
        task_index = path[1]
        task_text = _text(nodes[task_index]["message"])
        matches = [stem for stem in STEMS if f"/{stem}.md" in task_text]
        if len(matches) != 1:
            raise ValueError("cleanup child branch has ambiguous shard ownership")
        successful_calls = []
        for position, node_index in enumerate(path[:-1]):
            node = nodes[node_index]
            code = _tool_code(node.get("message") or {})
            if code is None or position + 1 >= len(path):
                continue
            receipt = nodes[path[position + 1]]
            receipt_text = _text(receipt.get("message") or {})
            if (
                receipt.get("message", {}).get("role") == "tool"
                and "Traceback" not in receipt_text
                and "deliveryStatus" in receipt_text
            ):
                successful_calls.append((node_index, path[position + 1], code))
        if len(successful_calls) != 1:
            raise ValueError("cleanup child branch lacks one successful compute-and-send action")
        action_index, receipt_index, code = successful_calls[0]
        if (
            "import json" not in code
            or "read_text" not in code
            or "len(" not in code
            or "receiver_role='parent'" not in code
        ):
            raise ValueError("cleanup child action is not the complete Python-first return program")
        rows.append(
            {
                "messages": [
                    _wire_message(runtime),
                    _wire_message(nodes[task_index]["message"]),
                    _wire_message(nodes[action_index]["message"]),
                    _wire_message(nodes[receipt_index]["message"]),
                    _wire_message(nodes[terminal_candidates[0]]["message"]),
                ],
                "tools": json.dumps(trace.get("tools") or [], sort_keys=True, separators=(",", ":")),
                "task_key": f"{trace['task']['data']['name']}:{matches[0]}",
                "trace_id": f"document-cleanup:child:{trace['id']}:{matches[0]}",
                "source_trace_id": trace["id"],
                "family": f"document_cleanup_child_{matches[0]}",
                "role": "child",
                "objective": OBJECTIVES["child"],
                "source_trace": str(source),
            }
        )
    if len(rows) != 3:
        raise ValueError("cleanup trace does not contain three successful child branches")
    return rows


def export(*, traces_path: Path, output_dir: Path, role: str) -> dict[str, Any]:
    if role not in SCHEMA_VERSIONS:
        raise ValueError("document cleanup role must be coordinator or child")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite document cleanup replay: {output_dir}")
    traces = []
    with traces_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            envelope = json.loads(line)
            traces.extend(envelope.get("traces") or [])
    if len(traces) != 4 or len({trace.get("id") for trace in traces}) != 4:
        raise ValueError("document cleanup requires four distinct source trajectories")
    for trace in traces:
        _admitted_trace(trace)
    traces.sort(key=lambda trace: trace["task"]["data"]["name"])
    if role == "coordinator":
        rows = _coordinator_rows(traces, traces_path.resolve())
    else:
        rows = [row for trace in traces for row in _child_rows(trace, traces_path.resolve())]
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 12 or len({row["trace_id"] for row in rows}) != 12:
        raise ValueError("document cleanup replay must contain twelve unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted({row["family"] for row in rows})
    }
    if set(family_counts.values()) != {4}:
        raise ValueError(f"document cleanup replay is not balanced: {family_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSIONS[role],
        "status": "complete",
        "role": role,
        "objective": OBJECTIVES[role],
        "rows": len(rows),
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "accepted_trace_ids": [trace["id"] for trace in traces],
        "source": {
            "traces_path": str(traces_path.resolve()),
            "traces_sha256": sha256_file(traces_path),
        },
        "successful_on_policy_sources_only": True,
        "projection": (
            "remove_failed_prefixes_and_emit_no_tool_partial_fanin_targets"
            if role == "coordinator"
            else "remove_failed_prefixes_and_replay_successful_compute_send_stop"
        ),
        "failed_prefixes_removed": True,
        "role_separated": True,
        "answer_free": role == "child",
        "environment_grounded": True,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(SCHEMA_VERSIONS), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces_path=args.traces.resolve(),
                output_dir=args.output_dir.resolve(),
                role=args.role,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
