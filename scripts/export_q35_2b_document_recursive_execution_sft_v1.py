#!/usr/bin/env python3
"""Build an answer-free SFT bootstrap for recursive document execution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from export_q35_2b_document_coordinator_fanin_sft_v1 import canonical_spawn_action
from export_q35_2b_document_decision_sft_v1 import (
    STEMS,
    _manager_instruction,
    _wire_message,
    canonical_first_action,
    sha256_file,
)

SCHEMA_VERSION = "qwen35-2b-document-recursive-execution-sft/v1"
OBJECTIVE = "answer_free_recursive_manager_admission_delegation_and_passive_yield"
PHASES = ("root_manager_admission", "manager_leaf_admission", "manager_passive_yield")


def _tool_schema(trace: dict[str, Any]) -> list[dict[str, Any]]:
    tools = trace.get("tools") or []
    if [tool.get("name") for tool in tools if isinstance(tool, dict)] != ["ipython"]:
        raise ValueError("recursive execution source must expose exactly the IPython tool")
    return tools


def _assistant_tool_call(
    *, reasoning: str, code: str, trace_id: str, phase: str
) -> tuple[dict[str, Any], str]:
    digest = hashlib.sha256(f"{trace_id}:{phase}:{code}".encode()).hexdigest()[:16]
    call_id = f"document-recursion-{digest}"
    return (
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
                        "arguments": json.dumps({"code": code}, separators=(",", ":")),
                    },
                }
            ],
        },
        call_id,
    )


def _manager_runtime(traces: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for trace in traces:
        for node in trace.get("nodes") or []:
            message = node.get("message") or {}
            content = message.get("content", "")
            if (
                node.get("parent") is None
                and node.get("sampled") is False
                and message.get("role") == "user"
                and isinstance(content, str)
                and "Recursive agent depth: 1" in content
                and "You are a child agent spawned by" in content
            ):
                candidates.append(_wire_message(message))
    if not candidates:
        raise ValueError("runtime sources contain no real depth-one agent context")
    return candidates[0]


def _hierarchical_task(trace: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    task = trace.get("task", {}).get("data", {})
    if task.get("family") != "document_hierarchical" or not isinstance(
        task.get("name"), str
    ):
        raise ValueError("recursive execution source is not a hierarchical document task")
    files = task.get("files")
    if not isinstance(files, dict) or len(files) != 3:
        raise ValueError("hierarchical document source lacks three fixtures")
    roots = {str(Path(path).parent) for path in files}
    if len(roots) != 1:
        raise ValueError("hierarchical document fixtures do not share one directory")
    root = roots.pop()
    if set(files) != {f"{root}/{stem}.md" for stem in STEMS}:
        raise ValueError("hierarchical document fixtures do not match the canonical shards")
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError("hierarchical document source lacks its root prompt")
    if (
        nodes[0].get("parent") is not None
        or nodes[0].get("sampled") is not False
        or nodes[1].get("parent") != 0
        or nodes[1].get("sampled") is not False
    ):
        raise ValueError("hierarchical document source lacks the canonical root prefix")
    return root, task["name"], [
        _wire_message(nodes[0]["message"]),
        _wire_message(nodes[1]["message"]),
    ]


def _rows(
    trace: dict[str, Any], *, source: Path, manager_runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    root, task_key, root_prefix = _hierarchical_task(trace)
    tools = _tool_schema(trace)
    trace_id = str(trace.get("id"))

    root_reasoning, root_code = canonical_first_action("document_hierarchical", root)
    root_action, _ = _assistant_tool_call(
        reasoning=root_reasoning,
        code=root_code,
        trace_id=trace_id,
        phase="root_manager_admission",
    )

    manager_prompt = {
        "role": "user",
        "content": f"[task from parent]\n{_manager_instruction(root)}",
    }
    manager_reasoning, manager_code = canonical_spawn_action(root)
    manager_reasoning = (
        "I am the non-root document coordinator. "
        + manager_reasoning.replace("I must", "I must", 1)
    )
    manager_action, manager_call_id = _assistant_tool_call(
        reasoning=manager_reasoning,
        code=manager_code,
        trace_id=trace_id,
        phase="manager_leaf_admission",
    )
    receipt = {
        "role": "tool",
        "content": (
            "alpha_worker=RLMSpawnHandle(name='alpha-document-worker')\n"
            "beta_worker=RLMSpawnHandle(name='beta-document-worker')\n"
            "gamma_worker=RLMSpawnHandle(name='gamma-document-worker')"
        ),
        "tool_call_id": manager_call_id,
    }
    passive = {
        "role": "assistant",
        "content": "All three terminal children are admitted; waiting for their explicit reports.",
        "reasoning_content": (
            "All required leaf admissions succeeded and their handles are retained. I must make "
            "no polling or messaging call now; ending this turn lets explicit child reports "
            "resume the manager session."
        ),
        "tool_calls": [],
    }

    phase_messages = {
        "root_manager_admission": [*copy.deepcopy(root_prefix), root_action],
        "manager_leaf_admission": [
            copy.deepcopy(manager_runtime),
            copy.deepcopy(manager_prompt),
            copy.deepcopy(manager_action),
        ],
        "manager_passive_yield": [
            copy.deepcopy(manager_runtime),
            copy.deepcopy(manager_prompt),
            copy.deepcopy(manager_action),
            receipt,
            passive,
        ],
    }
    return [
        {
            "messages": phase_messages[phase],
            "tools": json.dumps(tools, sort_keys=True, separators=(",", ":")),
            "task_key": f"{task_key}:{phase}",
            "trace_id": f"document-recursive-execution:{trace_id}:{phase}",
            "family": f"document_recursive_{phase}",
            "phase": phase,
            "role": "coordinator",
            "objective": OBJECTIVE,
            "source_trace": str(source),
        }
        for phase in PHASES
    ]


def export(
    *, traces: list[Path], runtime_traces: list[Path], output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite recursive execution bootstrap: {output_dir}")
    source_records = []
    hierarchical: list[tuple[dict[str, Any], Path]] = []
    for path in traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                for trace in (json.loads(line).get("traces") or []):
                    if trace.get("task", {}).get("data", {}).get("family") == "document_hierarchical":
                        hierarchical.append((trace, path.resolve()))
    if len(hierarchical) != 4:
        raise ValueError(f"recursive execution bootstrap requires four hierarchical tasks, found {len(hierarchical)}")

    runtime_records = []
    runtime_sources = []
    for path in runtime_traces:
        if not path.is_file():
            raise FileNotFoundError(path)
        runtime_records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    runtime_sources.extend(json.loads(line).get("traces") or [])
    runtime = _manager_runtime(runtime_sources)

    rows = []
    for trace, source in hierarchical:
        rows.extend(_rows(trace, source=source, manager_runtime=runtime))
    rows.sort(key=lambda row: row["task_key"])
    if len(rows) != 12 or len({row["task_key"] for row in rows}) != 12:
        raise ValueError("recursive execution bootstrap must contain twelve unique rows")
    family_counts = {
        family: sum(row["family"] == family for row in rows)
        for family in sorted({row["family"] for row in rows})
    }
    if set(family_counts.values()) != {4}:
        raise ValueError(f"recursive execution bootstrap is not phase-balanced: {family_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": len(rows),
        "family_counts": family_counts,
        "task_keys": [row["task_key"] for row in rows],
        "source_traces": source_records,
        "runtime_traces": runtime_records,
        "answer_free": True,
        "topology_choice_targeted": False,
        "child_policy_targeted": False,
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
    parser.add_argument("--runtime-traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                traces=[path.resolve() for path in args.traces],
                runtime_traces=[path.resolve() for path in args.runtime_traces],
                output_dir=args.output_dir.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
