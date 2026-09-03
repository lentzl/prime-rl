#!/usr/bin/env python3
"""Probe a separate specialist-router policy without executing workers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dual_policy_openai_proxy_v1 import specialist_manager_contract_from_messages
from export_q35_2b_adaptive_cognition_sft_v1 import _runtime_messages
from probe_q35_2b_split_specialist_routing_v1 import (
    EXPERT_IDS,
    _expert_payload,
    _expert_phase_messages,
    _manager_messages,
    _request,
    _root_messages,
    _selector_seed,
)
from subagent_communication_v1.taskset import (
    SPECIALIST_FAMILIES,
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
)

SCHEMA_VERSION = "qwen35-2b-specialist-router-policy-probe/v1"
FROZEN_SCREENS = {
    37700: 20261209,
    37800: 20261210,
    38000: 20261212,
    38200: 20261213,
    38300: 20261214,
    38400: 20261215,
}


def _root_expert(family: str) -> str | None:
    if family == "specialist_generic":
        return "generic_worker"
    if family.startswith("specialist_table_"):
        return "table_analyst"
    if family.startswith("specialist_source_"):
        return "source_inspector"
    return None


def probe(args: argparse.Namespace) -> dict[str, Any]:
    runtime, runtime_sources = _runtime_messages(args.runtime_traces)
    taskset = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="eval",
            families=tuple(SPECIALIST_FAMILIES),
            instances_per_template=1,
            instance_offset=args.instance_offset,
            seed=args.seed,
            available_experts=EXPERT_IDS,
        )
    )
    tasks = taskset.load()
    if len(tasks) != 16:
        raise ValueError(f"specialist-router probe requires 16 tasks, found {len(tasks)}")

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    rows = []
    with httpx.Client(
        timeout=args.timeout, headers={"Authorization": "Bearer local"}
    ) as client:
        for task in tasks:
            family = task.data.family
            expected = _root_expert(family)
            if expected is not None:
                messages = _root_messages(runtime[0], task.data.prompt)
                result = _request(
                    client,
                    endpoint=endpoint,
                    payload=_expert_payload(
                        model=args.model,
                        messages=_expert_phase_messages(messages),
                        seed=_selector_seed(f"{task.data.name}:router", args.seed),
                    ),
                    tool_name="select_expert",
                    field="expert_id",
                )
                rows.append(
                    {
                        "task": task.data.name,
                        "family": family,
                        "role_scope": "root",
                        "expected": expected,
                        **result,
                        "correct": result["selected"] == expected,
                    }
                )
            if family.startswith("specialist_recursive_"):
                manager = specialist_manager_contract_from_messages(
                    [{"role": "user", "content": task.data.prompt}]
                )
                if manager is None or task.data.preferred_expert not in EXPERT_IDS:
                    raise ValueError(f"invalid recursive specialist task: {task.data.name}")
                messages = _manager_messages(runtime[1], manager)
                result = _request(
                    client,
                    endpoint=endpoint,
                    payload=_expert_payload(
                        model=args.model,
                        messages=_expert_phase_messages(messages),
                        seed=_selector_seed(f"{task.data.name}:manager-router", args.seed),
                    ),
                    tool_name="select_expert",
                    field="expert_id",
                )
                rows.append(
                    {
                        "task": f"{task.data.name}:specialist-manager",
                        "family": family,
                        "role_scope": "nonroot_specialist_manager",
                        "expected": task.data.preferred_expert,
                        **result,
                        "correct": result["selected"] == task.data.preferred_expert,
                    }
                )

    root_rows = [row for row in rows if row["role_scope"] == "root"]
    manager_rows = [row for row in rows if row["role_scope"] != "root"]
    root_correct = Counter(row["expected"] for row in root_rows if row["correct"])
    selected = Counter(row["selected"] for row in rows if row["selected"] is not None)
    maximum_share = max(selected.values(), default=0) / len(rows)
    normalizable = all(row["normalizable_one_field_transport"] for row in rows)
    acceptance = {
        "minimum_correct_root_experts": sum(row["correct"] for row in root_rows) >= 7,
        "minimum_correct_generic_experts": root_correct["generic_worker"] >= 1,
        "minimum_correct_table_experts": root_correct["table_analyst"] >= 3,
        "minimum_correct_source_experts": root_correct["source_inspector"] >= 3,
        "minimum_correct_recursive_manager_experts": sum(
            row["correct"] for row in manager_rows
        )
        >= 3,
        "normalizable_exact_one_field_transport": normalizable,
        "maximum_selected_expert_share": maximum_share <= 0.75,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "optimizer_updates": args.model_optimizer_updates,
        "instance_offset": args.instance_offset,
        "seed": args.seed,
        "runtime_sources": runtime_sources,
        "root_expert_correct": sum(row["correct"] for row in root_rows),
        "root_expert_total": len(root_rows),
        "root_correct_by_expert": dict(sorted(root_correct.items())),
        "root_total_by_expert": dict(
            sorted(Counter(row["expected"] for row in root_rows).items())
        ),
        "recursive_manager_expert_correct": sum(
            row["correct"] for row in manager_rows
        ),
        "recursive_manager_expert_total": len(manager_rows),
        "selected_expert_counts": dict(sorted(selected.items())),
        "maximum_selected_expert_share": maximum_share,
        "normalizable_one_field_transport": normalizable,
        "raw_function_tool_call_transport": all(
            row["exact_one_field_typed"] for row in rows
        ),
        "rows": rows,
        "acceptance_gates_relaxed": False,
        "acceptance": acceptance,
        "expert_router_probe_passed": all(acceptance.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8101/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-traces", action="append", type=Path, required=True)
    parser.add_argument("--instance-offset", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-optimizer-updates", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if FROZEN_SCREENS.get(args.instance_offset) != args.seed:
        parser.error("specialist-router probe offset and seed are frozen")
    if args.model_optimizer_updates not in {0, 1, 4, 8}:
        parser.error("router optimizer updates must be zero, one, four, or eight")
    args.runtime_traces = [path.resolve() for path in args.runtime_traces]
    result = probe(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite router result: {args.output}")
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
