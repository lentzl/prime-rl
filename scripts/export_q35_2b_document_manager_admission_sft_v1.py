#!/usr/bin/env python3
"""Build an exact-context SFT bootstrap for depth-one manager leaf admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_coordinator_fanin_sft_v1 import canonical_spawn_action
from export_q35_2b_document_decision_sft_v1 import STEMS, _wire_message, sha256_file

SCHEMA_VERSION = "qwen35-2b-document-manager-admission-sft/v1"
OBJECTIVE = "answer_free_depth_two_manager_leaf_admission"
FAMILY = "document_recursive_manager_leaf_admission"
MANAGER_CONTRACT = "[recursive document coordinator session contract]"


def _text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text" for part in content):
        return "".join(str(part.get("text", "")) for part in content)
    raise ValueError("manager source contains a non-text message")


def _manager_prefix(trace: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("manager source has no node graph")
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for root_index, root_node in enumerate(nodes):
        root_message = root_node.get("message") or {}
        if not (
            root_node.get("parent") is None and root_node.get("sampled") is False and root_message.get("role") == "user"
        ):
            continue
        root_text = _text(root_message)
        if not ("Recursive agent depth: 1" in root_text and "You are a child agent spawned by" in root_text):
            continue
        for task_node in nodes:
            task_message = task_node.get("message") or {}
            if not (
                task_node.get("parent") == root_index
                and task_node.get("sampled") is False
                and task_message.get("role") == "user"
            ):
                continue
            task_text = _text(task_message)
            if (
                "[task from parent]" in task_text
                and MANAGER_CONTRACT in task_text
                and "session_role=document_coordinator" in task_text
            ):
                matches.append((root_message, task_message))
    if len(matches) != 1:
        raise ValueError(f"manager source must contain exactly one live depth-one manager prefix, found {len(matches)}")
    return [_wire_message(message) for message in matches[0]]


def _assistant_action(*, root: str, trace_id: str) -> dict[str, Any]:
    reasoning, code = canonical_spawn_action(root)
    call_id = hashlib.sha256(f"{trace_id}:{code}".encode()).hexdigest()[:16]
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I am the non-root document coordinator. " + reasoning,
        "tool_calls": [
            {
                "id": f"document-manager-admission-{call_id}",
                "type": "function",
                "function": {
                    "name": "ipython",
                    "arguments": json.dumps({"code": code}, separators=(",", ":")),
                },
            }
        ],
    }


def _row(trace: dict[str, Any], *, source: Path) -> dict[str, Any]:
    task = trace.get("task", {}).get("data", {})
    if task.get("family") != "document_hierarchical" or not isinstance(task.get("name"), str):
        raise ValueError("manager admission source is not a hierarchical document task")
    files = task.get("files")
    if not isinstance(files, dict) or len(files) != 3:
        raise ValueError("manager admission source lacks three document fixtures")
    roots = {str(Path(path).parent) for path in files}
    if len(roots) != 1:
        raise ValueError("manager admission fixtures do not share one directory")
    root = roots.pop()
    if set(files) != {f"{root}/{stem}.md" for stem in STEMS}:
        raise ValueError("manager admission fixtures do not match canonical shards")
    tools = trace.get("tools") or []
    if [tool.get("name") for tool in tools if isinstance(tool, dict)] != ["ipython"]:
        raise ValueError("manager admission source must expose exactly the IPython tool")
    trace_id = str(trace.get("id"))
    return {
        "messages": [*_manager_prefix(trace), _assistant_action(root=root, trace_id=trace_id)],
        "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
        "task_key": task["name"],
        "trace_id": f"document-manager-admission:{trace_id}",
        "family": FAMILY,
        "phase": "manager_leaf_admission",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "source_trace": str(source),
    }


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite manager admission bootstrap: {output_dir}")
    rows: list[dict[str, Any]] = []
    sources = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        sources.append({"path": str(resolved), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    for trace in json.loads(line).get("traces") or []:
                        if trace.get("task", {}).get("data", {}).get("family") == "document_hierarchical":
                            rows.append(_row(trace, source=resolved))
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 4 or len({row["task_key"] for row in rows}) != 4:
        raise ValueError("manager admission bootstrap requires four unique live contexts")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": 4,
        "family_counts": {FAMILY: 4},
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": sources,
        "answer_free": True,
        "exact_live_depth_two_context": True,
        "root_policy_targeted": False,
        "child_policy_targeted": False,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                output_dir=args.output_dir.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
