#!/usr/bin/env python3
"""Build balanced exact-context SFT for level-invariant cognition decisions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from dual_policy_openai_proxy_v1 import (
    ROOT_COORDINATOR_CONTRACT,
    cognitive_action_from_facts,
    force_typed_cognitive_action_schema,
    local_cognition_facts_from_messages,
)
from export_q35_2b_document_decision_sft_v1 import _wire_message, sha256_file
from subagent_communication_v1.taskset import (
    ADAPTIVE_DOCUMENT_DEPTHS,
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
    _document_depth3_manager_instruction,
    _document_manager_instruction,
    _document_subgroup_manager_instruction,
)

SCHEMA_VERSION = "qwen35-2b-adaptive-cognition-sft/v1"
OBJECTIVE = "answer_free_level_invariant_local_cognition_action"
ACTIONS = ("solve_owned", "delegate_terminal", "delegate_coordinator")
FAMILIES = tuple(ADAPTIVE_DOCUMENT_DEPTHS)
ROWS_PER_ACTION = 16
ROWS = ROWS_PER_ACTION * len(ACTIONS)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") == "text" for part in content
    ):
        return "".join(str(part.get("text", "")) for part in content)
    raise ValueError("adaptive cognition runtime contains non-text content")


def _runtime_messages(paths: list[Path]) -> tuple[dict[int, dict[str, Any]], list[dict[str, str]]]:
    by_depth: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        sources.append({"path": str(resolved), "sha256": sha256_file(resolved)})
        with resolved.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                for trace in json.loads(line).get("traces") or []:
                    for node in trace.get("nodes") or []:
                        message = node.get("message") or {}
                        if not (
                            node.get("parent") is None
                            and node.get("sampled") is False
                            and message.get("role") == "user"
                        ):
                            continue
                        text = _message_text(message)
                        for depth in (0, 1, 2):
                            if f"Recursive agent depth: {depth}" in text:
                                by_depth.setdefault(depth, _wire_message(message))
    if set(by_depth) != {0, 1, 2}:
        raise ValueError(
            f"adaptive cognition runtime sources lack depths 0, 1, and 2: {sorted(by_depth)}"
        )
    return by_depth, sources


def _typed_tools() -> list[dict[str, Any]]:
    payload = force_typed_cognitive_action_schema(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ipython",
                        "description": "Execute Python in the persistent kernel.",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }
    )
    return [copy.deepcopy(payload["tools"][0]["function"])]


def _target(*, facts: dict[str, bool], action: str, key: str) -> dict[str, Any]:
    digest = hashlib.sha256(f"{key}:{action}".encode()).hexdigest()[:16]
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": (
            "I will apply the public local cognition facts to choose the cheapest "
            "sufficient next action for this session."
        ),
        "tool_calls": [
            {
                "id": f"adaptive-cognition-{digest}",
                "type": "function",
                "function": {
                    "name": "select_cognitive_action",
                    "arguments": json.dumps(
                        {**facts, "action": action}, separators=(",", ":")
                    ),
                },
            }
        ],
    }


def _row(
    *,
    runtime: dict[str, Any],
    prompt: str,
    key: str,
    depth: int,
    role_scope: str,
    root: bool,
) -> dict[str, Any]:
    facts = local_cognition_facts_from_messages([{"role": "user", "content": prompt}])
    if facts is None:
        raise ValueError(f"adaptive cognition prompt lacks facts: {key}")
    action = cognitive_action_from_facts(facts)
    prefix = copy.deepcopy(runtime)
    if root:
        prefix = {
            "role": "system",
            "content": f"{ROOT_COORDINATOR_CONTRACT}\n\n{_message_text(prefix).strip()}",
        }
    messages = [
        prefix,
        {"role": "user", "content": prompt if root else f"[task from parent]\n{prompt}"},
        _target(facts=facts, action=action, key=key),
    ]
    return {
        "messages": messages,
        "tools": json.dumps(_typed_tools(), sort_keys=True, separators=(",", ":")),
        "task_key": key,
        "trace_id": f"adaptive-cognition:{key}",
        "family": f"adaptive_{action}",
        "action": action,
        "coordination_depth": depth,
        "role_scope": role_scope,
        "role": "coordinator",
        "objective": OBJECTIVE,
    }


def _root_path(task: Any) -> str:
    roots = {str(Path(path).parent) for path in task.data.files}
    if len(roots) != 1:
        raise ValueError("adaptive task fixtures do not share one root")
    return roots.pop()


def _candidate_rows(runtime: dict[int, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    taskset = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=FAMILIES,
            instances_per_template=2,
            instance_offset=34100,
            seed=20261113,
        )
    )
    tasks = taskset.load()
    roots = {task.data.family: [] for task in tasks}
    for task in tasks:
        family = task.data.family
        depth = ADAPTIVE_DOCUMENT_DEPTHS[family]
        roots[family].append(
            _row(
                runtime=runtime[0],
                prompt=task.data.prompt,
                key=task.data.name,
                depth=depth,
                role_scope="root",
                root=True,
            )
        )

    d2_tasks = [task for task in tasks if task.data.family == "document_adaptive_d2"]
    d3_tasks = [task for task in tasks if task.data.family == "document_adaptive_d3"]
    d2_managers = [
        _row(
            runtime=runtime[1],
            prompt=_document_manager_instruction(_root_path(task)),
            key=f"{task.data.name}:manager",
            depth=1,
            role_scope="nonroot_manager",
            root=False,
        )
        for task in d2_tasks
    ]
    d3_tops = [
        _row(
            runtime=runtime[1],
            prompt=_document_depth3_manager_instruction(_root_path(task)),
            key=f"{task.data.name}:top-manager",
            depth=2,
            role_scope="nonroot_manager",
            root=False,
        )
        for task in d3_tasks
    ]
    d3_subgroups = [
        _row(
            runtime=runtime[2],
            prompt=_document_subgroup_manager_instruction(
                _root_path(task), group, stems
            ),
            key=f"{task.data.name}:{group}-manager",
            depth=1,
            role_scope="nonroot_subgroup_manager",
            root=False,
        )
        for task in d3_tasks
        for group, stems in (("alpha,beta", ("alpha", "beta")), ("gamma", ("gamma",)))
    ]

    solve = []
    for source_row in roots["document_adaptive_d0"]:
        for repeat in (1, 2):
            row = copy.deepcopy(source_row)
            row["task_key"] += f":solve-anchor-{repeat}"
            row["trace_id"] = f"adaptive-cognition:{row['task_key']}"
            solve.append(row)
    terminal = [
        *roots["document_adaptive_d1"],
        *d2_managers[:4],
        *d3_subgroups[:4],
    ]
    coordinator = [
        *roots["document_adaptive_d2"][:4],
        *roots["document_adaptive_d3"][:4],
        *d3_tops,
    ]
    pools = {
        "solve_owned": solve,
        "delegate_terminal": terminal,
        "delegate_coordinator": coordinator,
    }
    if any(len(rows) != ROWS_PER_ACTION for rows in pools.values()):
        raise ValueError(
            f"adaptive cognition candidate pools are not balanced: "
            f"{ {action: len(rows) for action, rows in pools.items()} }"
        )
    return pools


def export(*, runtime_traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite adaptive cognition SFT: {output_dir}")
    runtime, sources = _runtime_messages(runtime_traces)
    pools = _candidate_rows(runtime)
    rows = [pools[action][index] for index in range(ROWS_PER_ACTION) for action in ACTIONS]
    if len(rows) != ROWS or len({row["task_key"] for row in rows}) != ROWS:
        raise ValueError("adaptive cognition SFT requires 48 unique rows")
    action_counts = {
        action: sum(row["action"] == action for row in rows) for action in ACTIONS
    }
    if set(action_counts.values()) != {ROWS_PER_ACTION}:
        raise ValueError(f"adaptive cognition SFT is not action-balanced: {action_counts}")
    for start in range(0, ROWS, 12):
        batch_counts = {
            action: sum(row["action"] == action for row in rows[start : start + 12])
            for action in ACTIONS
        }
        if set(batch_counts.values()) != {4}:
            raise ValueError(f"adaptive cognition batch is not balanced: {batch_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": ROWS,
        "training_batch_size": 16,
        "family_counts": {f"adaptive_{key}": value for key, value in action_counts.items()},
        "action_counts": action_counts,
        "task_keys": [row["task_key"] for row in rows],
        "runtime_traces": sources,
        "answer_free": True,
        "model_visible_topology_labels": False,
        "level_invariant_action_contract": True,
        "root_and_nonroot_coordinator_rows": True,
        "heldout_variants": [4, 5],
        "training_variants": [0, 1, 2, 3],
        "tool_call_format": "openai_function_v1",
        "dataset": {"path": parquet.name, "sha256": sha256_file(parquet)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-traces", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                runtime_traces=[path.resolve() for path in args.runtime_traces],
                output_dir=args.output_dir.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
