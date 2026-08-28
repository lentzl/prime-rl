#!/usr/bin/env python3
"""Compile training-only environment scaffolds for the 2B bootstrap curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from procedural_harness_master_v1.taskset import (
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterTaskset,
)

SCHEMA_VERSION = "qwen35-2b-environment-bootstrap-context/v1"
DEFAULT_AXES = (
    ("natural_n1a", 4_004_000),
    ("natural_n1a_local", 4_006_000),
    ("natural_n1b", 4_005_000),
    ("natural_direct_control", 4_007_000),
)
LEAK_LADDER = (
    "action_scaffold",
    "child_contract_scaffold",
    "spawn_contract_scaffold",
    "ownership_scaffold",
    "strategy_hint",
)
LeakLevel = Literal[
    "action_scaffold",
    "child_contract_scaffold",
    "spawn_contract_scaffold",
    "ownership_scaffold",
    "strategy_hint",
    "solution_replay",
]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _local_expression(path: str, family: str) -> tuple[list[str], str]:
    quoted_path = repr(path)
    if family == "json_sum":
        return ["import json"], f"sum(json.loads(Path({quoted_path}).read_text()))"
    if family == "csv_total":
        return ["import csv"], (
            f"sum(int(row['amount']) for row in csv.DictReader(Path({quoted_path}).read_text().splitlines()))"
        )
    if family == "word_count":
        raise ValueError("word_count needs its task-specific keyword and is not a local rung family")
    if family == "md_h2":
        return [], (f"sum(line.startswith('## ') for line in Path({quoted_path}).read_text().splitlines())")
    if family == "log_error":
        return [], (f"sum(line.startswith('ERROR ') for line in Path({quoted_path}).read_text().splitlines())")
    if family == "python_defs":
        return ["import ast"], (
            "sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in "
            f"ast.parse(Path({quoted_path}).read_text()).body)"
        )
    if family == "json_max":
        return ["import json"], (f"max(json.loads(Path({quoted_path}).read_text()).values())")
    if family == "tsv_total":
        return ["import csv"], (
            "sum(int(row['score']) for row in csv.DictReader("
            f"Path({quoted_path}).read_text().splitlines(), delimiter='\\t'))"
        )
    if family == "xml_items":
        return ["import xml.etree.ElementTree as ET"], (
            f"len(ET.fromstring(Path({quoted_path}).read_text()).findall('item'))"
        )
    if family == "jsonl_active_sum":
        return ["import json"], (
            "sum(row['value'] for row in (json.loads(line) for line in "
            f"Path({quoted_path}).read_text().splitlines()) if row['active'])"
        )
    if family == "ini_quota_sum":
        return ["import configparser"], (
            "(lambda parser: sum(parser.getint(section, 'quota') for section in "
            "parser.sections()))((lambda parser: (parser.read_string("
            f"Path({quoted_path}).read_text()), parser)[1])(configparser.ConfigParser()))"
        )
    raise ValueError(f"unsupported coordinator-local resource family: {family}")


def _child_instruction(child: dict[str, Any], leak_level: LeakLevel) -> str:
    expected = child["expected_result"]
    if leak_level == "solution_replay":
        reporting = (
            f"In your first IPython call execute exactly: await agent_message.send("
            f"{str(expected)!r}, receiver_role='parent')."
        )
    elif leak_level == "action_scaffold":
        reporting = (
            "Process the supplied private evidence and "
            f"{child['operation']}. In your first IPython call send only the resulting "
            "integer as a string to receiver_role='parent'."
        )
    else:
        reporting = (
            f"Process the supplied private evidence and {child['operation']}. Report only "
            "the resulting integer to your parent through the explicit message channel."
        )
    return (
        f"You are {child['name']}. Only you own {child['resource_path']}. {reporting} "
        "Send exactly once, then stop; do not return the result only as prose."
    )


def _bootstrap_context(task: Any, leak_level: LeakLevel) -> str:
    oracle = task.data.oracle
    final_answer = oracle["final_answer"]
    if task.data.family == "natural_direct_control":
        replay = ""
        if leak_level == "solution_replay":
            replay = f" The known successful terminal response is `{json.dumps(final_answer, separators=(',', ':'))}`."
        return (
            "This is an early-curriculum training episode. Do not delegate and do not call "
            "a tool. Perform the stated arithmetic and immediately return the requested bare "
            f"JSON.{replay}"
        )

    child = oracle["children"][0]
    imports: list[str] = []
    statements = []
    persistence = oracle.get("persistence_lease")
    if isinstance(persistence, dict):
        imports.append("from pathlib import Path")
        imports.append("import json")
        statements.append(f"captured = json.loads(Path({persistence['path']!r}).read_text())[{persistence['key']!r}]")

    child_prompt = _child_instruction(child, leak_level)
    statements.append(f"reviewer = await rlm({child_prompt!r}, name={child['name']!r})")

    local_variable = None
    for path, ownership in oracle.get("resource_ownership", {}).items():
        if ownership["owner"] != "coordinator":
            continue
        imports.append("from pathlib import Path")
        extra_imports, expression = _local_expression(path, ownership["family"])
        imports.extend(extra_imports)
        local_variable = "local"
        statements.append(f"{local_variable} = {expression}")

    code = "\n".join([*dict.fromkeys(imports), *statements])
    available = ["the delivered reviewer integer"]
    if local_variable is not None:
        available.append("the retained `local` value")
    if isinstance(persistence, dict):
        available.append("the retained `captured` value")

    replay = ""
    if leak_level == "solution_replay":
        replay = (
            " If recovery is needed after a failed attempt, the successful terminal response "
            f"is `{json.dumps(final_answer, separators=(',', ':'))}`."
        )
    if leak_level == "strategy_hint":
        return (
            "This is a training-only strategy hint. Respect the task's resource ownership: "
            "delegate the child-owned evidence once, preserve the admitted child handle, and "
            "stop taking actions when only the child report is outstanding. Resume from the "
            "delivered report and return only the requested bare JSON."
        )
    if leak_level == "ownership_scaffold":
        return (
            f"This is a training-only ownership scaffold. Child {child['name']!r} alone owns "
            f"{child['resource_path']!r} and must {child['operation']}; the root must not read "
            "that path. Spawn that child once with an instruction to report its integer through "
            "agent_message.send to the parent. Preserve the handle, complete any root-owned work, "
            "then end the turn without polling. Resume from the delivered message and return bare JSON."
        )
    if leak_level == "spawn_contract_scaffold":
        return (
            f"This is a training-only spawn contract. In the root's first IPython action, create "
            f"exactly one retained child named {child['name']!r}. Its instruction must identify "
            f"{child['resource_path']!r} as child-owned, ask it to {child['operation']}, and require "
            "one integer report through agent_message.send(..., receiver_role='parent'). Do not "
            "inspect the child path or await, poll, sleep, or list agents. Finish root-owned work, "
            "end the turn briefly, and resume only from the delivered report."
        )

    action_label = (
        "showing the exact next root action"
        if leak_level in {"action_scaffold", "solution_replay"}
        else "showing an exact root action with a less scripted child contract"
    )
    return (
        f"This is an early-curriculum training episode. The environment is deliberately {action_label}. "
        "In the root coordinator's first IPython call, execute "
        "this code exactly:\n\n"
        f"```python\n{code}\n```\n\n"
        "After that single call returns, do not call another tool, inspect the reviewer path, "
        "sleep, poll, list agents, or await the handle. End the turn with one short waiting "
        "sentence so the reviewer report can be delivered. When it arrives, use "
        + ", ".join(available)
        + " to compute the formula in the original request and return only its bare JSON."
        + replay
    )


def _parse_axis(value: str) -> tuple[str, int]:
    name, separator, start = value.partition(":")
    if not separator or not name or not start.isdigit():
        raise argparse.ArgumentTypeError("axis must have the form NAME:START_INDEX")
    return name, int(start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", action="append", type=_parse_axis)
    parser.add_argument("--tasks-per-axis", type=int, default=1)
    parser.add_argument("--master-seed", type=int, default=20260819)
    parser.add_argument(
        "--leak-level",
        choices=(*LEAK_LADDER, "solution_replay"),
        default="action_scaffold",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap artifact: {args.output}")
    if args.tasks_per_axis < 1:
        raise ValueError("tasks-per-axis must be positive")

    contexts: dict[str, str] = {}
    records = []
    axes = tuple(args.axis or DEFAULT_AXES)
    for axis, start_index in axes:
        tasks = ProceduralHarnessMasterTaskset(
            ProceduralHarnessMasterConfig(
                split="train_gen",
                count=args.tasks_per_axis,
                start_index=start_index,
                master_seed=args.master_seed,
                curriculum_rung=axis,
                private_payload_mode="finding_card",
            )
        ).load()
        for task in tasks:
            context = _bootstrap_context(task, args.leak_level)
            contexts[task.key] = context
            records.append(
                {
                    "episode_id": task.key,
                    "family": task.data.family,
                    "context_sha256": _sha256_text(context),
                    "final_answer_in_context": (args.leak_level == "solution_replay"),
                }
            )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "split": "train_gen",
        "curriculum_phase": "early_environment_scaffolding",
        "leak_level": args.leak_level,
        "leak_stage_index": (LEAK_LADDER.index(args.leak_level) if args.leak_level in LEAK_LADDER else None),
        "leak_ladder": list(LEAK_LADDER),
        "master_seed": args.master_seed,
        "private_payload_mode": "finding_card",
        "tasks_per_axis": args.tasks_per_axis,
        "axes": [{"name": name, "start_index": start} for name, start in axes],
        "heldout_allowed": False,
        "gradient_updates": 0,
        "contexts": contexts,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
