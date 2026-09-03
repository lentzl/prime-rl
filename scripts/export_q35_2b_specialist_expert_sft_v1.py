#!/usr/bin/env python3
"""Build balanced answer-free SFT for split specialist expert selection."""

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
    SPECIALIST_EXPERT_DECISION_PROMPT,
    force_typed_specialist_expert_schema,
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

SCHEMA_VERSION = "qwen35-2b-specialist-expert-sft/v1"
OBJECTIVE = "answer_free_public_registry_expert_only_selection"
EXPERT_IDS = tuple(SPECIALIST_EXPERTS)
ROWS_PER_EXPERT = 16
ROWS = len(EXPERT_IDS) * ROWS_PER_EXPERT
TRAINING_INSTANCE_OFFSET = 37600
TRAINING_SEED = 20261208


def _typed_tools() -> list[dict[str, Any]]:
    payload = force_typed_specialist_expert_schema(
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


def _target(*, expert_id: str, key: str) -> dict[str, Any]:
    digest = hashlib.sha256(f"{key}:{expert_id}".encode()).hexdigest()[:16]
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": (
            "I will use only the public capability registry and complete assignment "
            "to select the cheapest sufficient terminal expert."
        ),
        "tool_calls": [
            {
                "id": f"specialist-expert-{digest}",
                "type": "function",
                "function": {
                    "name": "select_expert",
                    "arguments": json.dumps(
                        {"expert_id": expert_id}, separators=(",", ":")
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
    expert_id: str,
    root: bool,
) -> dict[str, Any]:
    prefix = copy.deepcopy(runtime)
    if root:
        prefix = {
            "role": "system",
            "content": f"{ROOT_COORDINATOR_CONTRACT}\n\n{_message_text(prefix).strip()}",
        }
    return {
        "messages": [
            prefix,
            {
                "role": "user",
                "content": prompt if root else f"[task from parent]\n{prompt}",
            },
            {"role": "user", "content": SPECIALIST_EXPERT_DECISION_PROMPT},
            _target(expert_id=expert_id, key=key),
        ],
        "tools": json.dumps(_typed_tools(), sort_keys=True, separators=(",", ":")),
        "task_key": key,
        "trace_id": f"specialist-expert:{key}",
        "family": "specialist_expert_selection",
        "expert_id": expert_id,
        "coordination_depth": depth,
        "role_scope": role_scope,
        "role": "coordinator",
        "objective": OBJECTIVE,
    }


def _root_tasks() -> dict[str, list[Any]]:
    taskset = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=tuple(SPECIALIST_FAMILIES),
            instances_per_template=4,
            instance_offset=TRAINING_INSTANCE_OFFSET,
            seed=TRAINING_SEED,
            available_experts=EXPERT_IDS,
        )
    )
    grouped: dict[str, list[Any]] = {family: [] for family in SPECIALIST_FAMILIES}
    for task in taskset.load():
        grouped[task.data.family].append(task)
    for values in grouped.values():
        values.sort(key=lambda task: task.data.name)
    return grouped


def _candidate_rows(
    runtime: dict[int, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    tasks = _root_tasks()
    generic = [
        _row(
            runtime=runtime[0],
            prompt=task.data.prompt,
            key=task.data.name,
            depth=0,
            role_scope="root",
            expert_id="generic_worker",
            root=True,
        )
        for task in tasks["specialist_generic"][:ROWS_PER_EXPERT]
    ]
    result = {"generic_worker": generic}
    for expert_id, terminal_families, recursive_family in (
        (
            "table_analyst",
            ("specialist_table_join", "specialist_table_reconcile"),
            "specialist_recursive_table",
        ),
        (
            "source_inspector",
            ("specialist_source_ast", "specialist_source_config"),
            "specialist_recursive_source",
        ),
    ):
        roots = [
            _row(
                runtime=runtime[0],
                prompt=task.data.prompt,
                key=task.data.name,
                depth=0,
                role_scope="root",
                expert_id=expert_id,
                root=True,
            )
            for family in terminal_families
            for task in tasks[family][:4]
        ]
        managers = []
        for task in tasks[recursive_family][:8]:
            manager = specialist_manager_contract_from_messages(
                [{"role": "user", "content": task.data.prompt}]
            )
            if manager is None:
                raise ValueError(f"recursive specialist task lacks manager: {task.data.name}")
            managers.append(
                _row(
                    runtime=runtime[1],
                    prompt=manager,
                    key=f"{task.data.name}:specialist-manager",
                    depth=1,
                    role_scope="nonroot_specialist_manager",
                    expert_id=expert_id,
                    root=False,
                )
            )
        result[expert_id] = [*roots, *managers]
    if any(len(values) != ROWS_PER_EXPERT for values in result.values()):
        raise ValueError(
            "specialist expert pools are not balanced: "
            f"{ {key: len(values) for key, values in result.items()} }"
        )
    return result


def export(*, runtime_traces: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite specialist expert SFT: {output_dir}")
    runtime, sources = _runtime_messages(runtime_traces)
    pools = _candidate_rows(runtime)
    rows = [
        pools[expert_id][index]
        for index in range(ROWS_PER_EXPERT)
        for expert_id in EXPERT_IDS
    ]
    if len(rows) != ROWS or len({row["task_key"] for row in rows}) != ROWS:
        raise ValueError("specialist expert SFT requires 48 unique rows")
    expert_counts = {
        expert_id: sum(row["expert_id"] == expert_id for row in rows)
        for expert_id in EXPERT_IDS
    }
    role_counts = {
        expert_id: {
            role: sum(
                row["expert_id"] == expert_id and row["role_scope"] == role
                for row in rows
            )
            for role in ("root", "nonroot_specialist_manager")
        }
        for expert_id in EXPERT_IDS
    }
    first_batch_counts = {
        expert_id: sum(row["expert_id"] == expert_id for row in rows[:12])
        for expert_id in EXPERT_IDS
    }
    if set(expert_counts.values()) != {ROWS_PER_EXPERT} or set(
        first_batch_counts.values()
    ) != {4}:
        raise ValueError("specialist expert curriculum is not balanced")
    expected_roles = {
        "generic_worker": {"root": 16, "nonroot_specialist_manager": 0},
        "table_analyst": {"root": 8, "nonroot_specialist_manager": 8},
        "source_inspector": {"root": 8, "nonroot_specialist_manager": 8},
    }
    if role_counts != expected_roles:
        raise ValueError(f"specialist expert role balance is invalid: {role_counts}")

    output_dir.mkdir(parents=True)
    parquet = output_dir / "train.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "role": "coordinator",
        "objective": OBJECTIVE,
        "rows": ROWS,
        "training_batch_size": 12,
        "expert_counts": expert_counts,
        "role_counts": role_counts,
        "first_batch_expert_counts": first_batch_counts,
        "runtime_traces": sources,
        "answer_free": True,
        "public_capability_registry_only": True,
        "expert_only_tool_arguments": True,
        "cognitive_action_labels_present": False,
        "root_and_nonroot_coordinator_rows": True,
        "training_instance_offset": TRAINING_INSTANCE_OFFSET,
        "training_template_variants": [0, 1, 2, 3],
        "heldout_template_variants_excluded": [4, 5],
        "observed_instance_offsets_excluded": [35100, 37100, 37200, 37300],
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
