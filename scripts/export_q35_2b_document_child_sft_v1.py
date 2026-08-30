#!/usr/bin/env python3
"""Build an answer-free document leaf compute-and-report SFT bootstrap."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file

SCHEMA_VERSION = "qwen35-2b-document-child-sft/v1"
OBJECTIVE = "canonical_answer_free_document_leaf_compute_report_stop"
STEMS = ("alpha", "beta", "gamma")
SOURCE_ROOT_PATTERN = re.compile(r"/workspace/document-recursion/v\d+-i\d+")


def canonical_child_action(path: str) -> tuple[str, str]:
    reasoning = (
        "I own one terminal document shard. I will read it with Python, compute both counts "
        "from the complete contents, send one JSON report to my parent, and stop."
    )
    code = f"""from pathlib import Path
import json
path = Path({path!r})
text = path.read_text()
result = {{
    'words': len(text.split()),
    'h2': sum(line.startswith('## ') for line in text.splitlines()),
}}
await agent_message.send(json.dumps(result), receiver_role='parent')"""
    return reasoning, code


def _text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _child_templates(trace: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("document child source trace has no nodes")
    templates: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for root_index, root_node in enumerate(nodes):
        if root_node.get("parent") is not None:
            continue
        root_text = _text(root_node.get("message") or {})
        if "Recursive agent depth: 1" not in root_text or "You are a child agent" not in root_text:
            continue
        task_nodes = [
            node
            for node in nodes
            if node.get("parent") == root_index
            and node.get("sampled") is False
            and node.get("message", {}).get("role") == "user"
        ]
        if len(task_nodes) != 1:
            raise ValueError("document child source branch lacks one task prompt")
        task_node = task_nodes[0]
        task_text = _text(task_node["message"])
        matches = [stem for stem in STEMS if f"/{stem}.md" in task_text]
        if len(matches) != 1 or matches[0] in templates:
            raise ValueError("document child source branch has ambiguous shard ownership")
        templates[matches[0]] = (root_node["message"], task_node["message"])
    if set(templates) != set(STEMS):
        raise ValueError(f"document child source lacks three leaf templates: {sorted(templates)}")
    return templates


def _replace_root(message: dict[str, Any], target_root: str) -> dict[str, Any]:
    result = copy.deepcopy(message)
    content = result.get("content")
    if isinstance(content, str):
        result["content"] = SOURCE_ROOT_PATTERN.sub(target_root, content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] = SOURCE_ROOT_PATTERN.sub(target_root, part["text"])
    else:
        raise ValueError("document child task prompt is not textual")
    return result


def _rows(trace: dict[str, Any], *, source: Path) -> list[dict[str, Any]]:
    task = trace.get("task", {}).get("data", {})
    if task.get("family") != "document_flat":
        raise ValueError("document child source must be a flat document trace")
    templates = _child_templates(trace)
    tools = trace.get("tools") or []
    if [tool.get("name") for tool in tools if isinstance(tool, dict)] != ["ipython"]:
        raise ValueError("document child source must expose exactly the IPython tool")

    rows = []
    for variant in range(4):
        root = f"/workspace/document-recursion/v{variant}-i20000"
        for stem in STEMS:
            path = f"{root}/{stem}.md"
            reasoning, code = canonical_child_action(path)
            digest = hashlib.sha256(f"{variant}:{stem}:{code}".encode()).hexdigest()[:16]
            call_id = f"document-child-{digest}"
            runtime_message, task_message = templates[stem]
            rows.append(
                {
                    "messages": [
                        _wire_message(runtime_message),
                        _wire_message(_replace_root(task_message, root)),
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": reasoning,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "ipython",
                                        "arguments": json.dumps(
                                            {"code": code}, separators=(",", ":")
                                        ),
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "content": "message queued",
                            "tool_call_id": call_id,
                        },
                        {
                            "role": "assistant",
                            "content": "Done.",
                            "reasoning_content": (
                                "The report was delivered successfully, so I must stop."
                            ),
                            "tool_calls": [],
                        },
                    ],
                    "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
                    "task_key": f"document-child-v{variant}-i20000-{stem}",
                    "trace_id": f"document-child:{trace.get('id')}:{variant}:{stem}",
                    "family": f"document_child_{stem}",
                    "role": "child",
                    "objective": OBJECTIVE,
                    "source_trace": str(source),
                }
            )
    return rows


def export(*, traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite document child bootstrap: {output_dir}")
    candidates = []
    source_records = []
    selected_trace: dict[str, Any] | None = None
    selected_source: Path | None = None
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                envelope = json.loads(line)
                for trace in envelope.get("traces") or []:
                    task = trace.get("task", {}).get("data", {})
                    roots = sum(node.get("parent") is None for node in trace.get("nodes") or [])
                    if task.get("family") == "document_flat" and roots >= 4:
                        candidates.append((trace, path.resolve()))
    if len(candidates) != 1:
        raise ValueError(f"expected one complete three-child template trace, found {len(candidates)}")
    selected_trace, selected_source = candidates[0]
    rows = _rows(selected_trace, source=selected_source)
    if len(rows) != 12 or len({row["task_key"] for row in rows}) != 12:
        raise ValueError("document child bootstrap must contain twelve unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted({row["family"] for row in rows})
    }
    if set(family_counts.values()) != {4}:
        raise ValueError(f"document child bootstrap is not shard-balanced: {family_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "child",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": source_records,
        "selected_trace_id": selected_trace.get("id"),
        "answer_free": True,
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
