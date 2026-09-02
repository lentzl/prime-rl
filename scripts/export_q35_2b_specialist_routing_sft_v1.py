#!/usr/bin/env python3
"""Build balanced answer-free SFT for public specialist routing decisions."""

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
    force_typed_specialist_action_schema,
    specialist_manager_contract_from_messages,
)
from export_q35_2b_adaptive_cognition_sft_v1 import (
    _message_text,
    _runtime_messages,
)
from export_q35_2b_document_decision_sft_v1 import sha256_file
from subagent_communication_v1.taskset import (
    SPECIALIST_EXPERTS,
    SPECIALIST_FAMILIES,
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
)

SCHEMA_VERSION = "qwen35-2b-specialist-routing-sft/v1"
OBJECTIVE = "answer_free_public_registry_action_and_expert_selection"
EXPERT_IDS = tuple(SPECIALIST_EXPERTS)
ROUTE_CLASSES = (
    "delegate_terminal:source_inspector",
    "delegate_terminal:generic_worker",
    "delegate_terminal:table_analyst",
    "solve_owned:none",
    "delegate_coordinator:none",
)
ROWS_PER_ROUTE = 16
ROWS = len(ROUTE_CLASSES) * ROWS_PER_ROUTE


def _typed_tools() -> list[dict[str, Any]]:
    payload = force_typed_specialist_action_schema(
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
        },
        EXPERT_IDS,
    )
    return [copy.deepcopy(payload["tools"][0]["function"])]


def _target(*, action: str, expert_id: str, key: str) -> dict[str, Any]:
    digest = hashlib.sha256(f"{key}:{action}:{expert_id}".encode()).hexdigest()[:16]
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": (
            "I will use the public local facts and capability descriptions to select the "
            "cheapest sufficient route without inspecting any hidden answer."
        ),
        "tool_calls": [
            {
                "id": f"specialist-routing-{digest}",
                "type": "function",
                "function": {
                    "name": "select_cognitive_action",
                    "arguments": json.dumps(
                        {"action": action, "expert_id": expert_id},
                        separators=(",", ":"),
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
    action: str,
    expert_id: str,
    root: bool,
) -> dict[str, Any]:
    prefix = copy.deepcopy(runtime)
    if root:
        prefix = {
            "role": "system",
            "content": f"{ROOT_COORDINATOR_CONTRACT}\n\n{_message_text(prefix).strip()}",
        }
    route_class = f"{action}:{expert_id}"
    return {
        "messages": [
            prefix,
            {
                "role": "user",
                "content": prompt if root else f"[task from parent]\n{prompt}",
            },
            _target(action=action, expert_id=expert_id, key=key),
        ],
        "tools": json.dumps(_typed_tools(), sort_keys=True, separators=(",", ":")),
        "task_key": key,
        "trace_id": f"specialist-routing:{key}",
        "family": "specialist_routing",
        "route_class": route_class,
        "action": action,
        "expert_id": expert_id,
        "coordination_depth": depth,
        "role_scope": role_scope,
        "role": "coordinator",
        "objective": OBJECTIVE,
    }


def _root_route(family: str, preferred_expert: str | None) -> tuple[str, str]:
    if family == "specialist_local":
        return "solve_owned", "none"
    if family.startswith("specialist_recursive_"):
        return "delegate_coordinator", "none"
    if preferred_expert not in EXPERT_IDS:
        raise ValueError(f"terminal specialist task lacks a public expert: {family}")
    return "delegate_terminal", preferred_expert


def _root_rows(runtime: dict[int, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    taskset = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=tuple(SPECIALIST_FAMILIES),
            instances_per_template=4,
            instance_offset=37000,
            seed=20261202,
            available_experts=EXPERT_IDS,
        )
    )
    rows: dict[str, list[dict[str, Any]]] = {
        family: [] for family in SPECIALIST_FAMILIES
    }
    for task in taskset.load():
        family = task.data.family
        action, expert_id = _root_route(family, task.data.preferred_expert)
        rows[family].append(
            _row(
                runtime=runtime[0],
                prompt=task.data.prompt,
                key=task.data.name,
                depth=0,
                role_scope="root",
                action=action,
                expert_id=expert_id,
                root=True,
            )
        )
    for values in rows.values():
        values.sort(key=lambda row: row["task_key"])
    return rows


def _manager_rows(
    runtime: dict[int, dict[str, Any]],
    roots: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for family, expert_id in (
        ("specialist_recursive_table", "table_analyst"),
        ("specialist_recursive_source", "source_inspector"),
    ):
        values = []
        for root in roots[family]:
            prompt = root["messages"][1]["content"]
            manager = specialist_manager_contract_from_messages(
                [{"role": "user", "content": prompt}]
            )
            if manager is None:
                raise ValueError(
                    f"recursive specialist row lacks a manager contract: {root['task_key']}"
                )
            values.append(
                _row(
                    runtime=runtime[1],
                    prompt=manager,
                    key=f"{root['task_key']}:specialist-manager",
                    depth=1,
                    role_scope="nonroot_specialist_manager",
                    action="delegate_terminal",
                    expert_id=expert_id,
                    root=False,
                )
            )
        result[family] = values
    return result


def _candidate_rows(
    runtime: dict[int, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    roots = _root_rows(runtime)
    managers = _manager_rows(runtime, roots)
    pools = {
        "solve_owned:none": roots["specialist_local"][:ROWS_PER_ROUTE],
        "delegate_terminal:generic_worker": roots["specialist_generic"][
            :ROWS_PER_ROUTE
        ],
        "delegate_terminal:table_analyst": [
            *roots["specialist_table_join"][:4],
            *roots["specialist_table_reconcile"][:4],
            *managers["specialist_recursive_table"][:8],
        ],
        "delegate_terminal:source_inspector": [
            *roots["specialist_source_ast"][:4],
            *roots["specialist_source_config"][:4],
            *managers["specialist_recursive_source"][:8],
        ],
        "delegate_coordinator:none": [
            *roots["specialist_recursive_table"][:8],
            *roots["specialist_recursive_source"][:8],
        ],
    }
    if any(len(values) != ROWS_PER_ROUTE for values in pools.values()):
        raise ValueError(
            "specialist routing pools are not balanced: "
            f"{ {key: len(values) for key, values in pools.items()} }"
        )
    return pools


def export(*, runtime_traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite specialist routing SFT: {output_dir}"
        )
    runtime, sources = _runtime_messages(runtime_traces)
    pools = _candidate_rows(runtime)
    rows = [
        pools[route_class][index]
        for index in range(ROWS_PER_ROUTE)
        for route_class in ROUTE_CLASSES
    ]
    if len(rows) != ROWS or len({row["task_key"] for row in rows}) != ROWS:
        raise ValueError("specialist routing SFT requires 80 unique rows")
    route_counts = {
        route_class: sum(row["route_class"] == route_class for row in rows)
        for route_class in ROUTE_CLASSES
    }
    first_batch_counts = {
        route_class: sum(row["route_class"] == route_class for row in rows[:16])
        for route_class in ROUTE_CLASSES
    }
    if set(route_counts.values()) != {ROWS_PER_ROUTE} or sorted(
        first_batch_counts.values()
    ) != [3, 3, 3, 3, 4]:
        raise ValueError("specialist routing curriculum is not balanced")

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
        "route_counts": route_counts,
        "first_batch_route_counts": first_batch_counts,
        "runtime_traces": sources,
        "answer_free": True,
        "public_capability_registry_only": True,
        "action_and_expert_only_tool_arguments": True,
        "root_and_nonroot_coordinator_rows": True,
        "training_template_variants": [0, 1, 2, 3],
        "heldout_template_variants_excluded": [4, 5],
        "observed_instance_offset_excluded": 35100,
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
