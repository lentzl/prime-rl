#!/usr/bin/env python3
"""Probe frozen coordinator routing with separate one-field typed decisions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dual_policy_openai_proxy_v1 import (
    ROOT_COORDINATOR_CONTRACT,
    force_typed_cognitive_action_schema,
    specialist_manager_contract_from_messages,
)
from export_q35_2b_adaptive_cognition_sft_v1 import (
    _message_text,
    _runtime_messages,
)
from subagent_communication_v1.taskset import (
    SPECIALIST_EXPERTS,
    SPECIALIST_FAMILIES,
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
)

SCHEMA_VERSION = "qwen35-2b-split-specialist-routing-probe/v1"
EXPERT_IDS = tuple(SPECIALIST_EXPERTS)


def _selector_seed(key: str, seed: int) -> int:
    return seed + int(hashlib.sha256(key.encode()).hexdigest()[:6], 16)


def _base_payload(
    *, model: str, messages: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "top_p": 0.95,
        "max_tokens": 256,
        "seed": seed,
        "stream": False,
    }


def _action_payload(
    *, model: str, messages: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    payload = _base_payload(model=model, messages=messages, seed=seed)
    payload["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "ipython",
                "description": "Execute Python in the persistent kernel.",
                "parameters": {"type": "object"},
            },
        }
    ]
    return force_typed_cognitive_action_schema(payload)


def _expert_payload(
    *, model: str, messages: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    payload = _base_payload(model=model, messages=messages, seed=seed)
    payload.update(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_expert",
                        "description": (
                            "The cognitive action is already delegate_terminal. Select exactly "
                            "one terminal worker from the public capability registry."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expert_id": {
                                    "type": "string",
                                    "enum": list(EXPERT_IDS),
                                }
                            },
                            "required": ["expert_id"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "select_expert"},
            },
            "parallel_tool_calls": False,
        }
    )
    return payload


def _arguments_from_response(
    response: dict[str, Any], *, tool_name: str, field: str
) -> tuple[str | None, bool, dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None, False, {}
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None, False, {}
    parsed = None
    typed = False
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) == 1:
        call = tool_calls[0]
        function = call.get("function") if isinstance(call, dict) else None
        if (
            isinstance(call, dict)
            and call.get("type") == "function"
            and isinstance(function, dict)
            and function.get("name") == tool_name
        ):
            arguments = function.get("arguments")
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                parsed = None
            typed = True
    if parsed is None:
        for key in ("content", "reasoning", "reasoning_content"):
            value = message.get(key)
            if not isinstance(value, str):
                continue
            try:
                candidate = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
    valid = isinstance(parsed, dict) and set(parsed) == {field} and isinstance(parsed[field], str)
    return (parsed[field] if valid else None), typed and valid, message


def _expected_root_route(family: str) -> tuple[str, str | None]:
    if family == "specialist_local":
        return "solve_owned", None
    if family.startswith("specialist_recursive_"):
        return "delegate_coordinator", None
    if family == "specialist_generic":
        return "delegate_terminal", "generic_worker"
    if family.startswith("specialist_table_"):
        return "delegate_terminal", "table_analyst"
    if family.startswith("specialist_source_"):
        return "delegate_terminal", "source_inspector"
    raise ValueError(f"unsupported specialist family: {family}")


def _root_messages(runtime: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": f"{ROOT_COORDINATOR_CONTRACT}\n\n{_message_text(runtime).strip()}",
        },
        {"role": "user", "content": prompt},
    ]


def _manager_messages(runtime: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(runtime),
        {"role": "user", "content": f"[task from parent]\n{prompt}"},
    ]


def _expert_phase_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    result.append(
        {
            "role": "user",
            "content": (
                "[split specialist decision phase]\n"
                "The legal cognitive action has already been fixed as delegate_terminal. "
                "Use only the public capability registry and complete assignment to select "
                "one expert_id. Do not solve the task or inspect any file."
            ),
        }
    )
    return result


def _request(
    client: httpx.Client,
    *,
    endpoint: str,
    payload: dict[str, Any],
    tool_name: str,
    field: str,
) -> dict[str, Any]:
    response = client.post(endpoint, json=payload)
    response.raise_for_status()
    body = response.json()
    selected, exact_typed, message = _arguments_from_response(
        body, tool_name=tool_name, field=field
    )
    return {
        "selected": selected,
        "exact_one_field_typed": exact_typed,
        "response_message": message,
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
    }


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
        raise ValueError(f"split routing probe requires 16 root tasks, found {len(tasks)}")

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    action_rows = []
    expert_rows = []
    with httpx.Client(timeout=args.timeout, headers={"Authorization": "Bearer local"}) as client:
        for task in tasks:
            family = task.data.family
            expected_action, expected_expert = _expected_root_route(family)
            messages = _root_messages(runtime[0], task.data.prompt)
            action = _request(
                client,
                endpoint=endpoint,
                payload=_action_payload(
                    model=args.model,
                    messages=messages,
                    seed=_selector_seed(f"{task.data.name}:action", args.seed),
                ),
                tool_name="select_cognitive_action",
                field="action",
            )
            action_rows.append(
                {
                    "task": task.data.name,
                    "family": family,
                    "expected": expected_action,
                    **action,
                    "correct": action["selected"] == expected_action,
                }
            )
            if expected_expert is not None:
                expert = _request(
                    client,
                    endpoint=endpoint,
                    payload=_expert_payload(
                        model=args.model,
                        messages=_expert_phase_messages(messages),
                        seed=_selector_seed(f"{task.data.name}:expert", args.seed),
                    ),
                    tool_name="select_expert",
                    field="expert_id",
                )
                expert_rows.append(
                    {
                        "task": task.data.name,
                        "family": family,
                        "role_scope": "root",
                        "expected": expected_expert,
                        **expert,
                        "correct": expert["selected"] == expected_expert,
                    }
                )
            if family.startswith("specialist_recursive_"):
                manager_prompt = specialist_manager_contract_from_messages(
                    [{"role": "user", "content": task.data.prompt}]
                )
                if manager_prompt is None or task.data.preferred_expert not in EXPERT_IDS:
                    raise ValueError(f"invalid recursive specialist task: {task.data.name}")
                manager_messages = _manager_messages(runtime[1], manager_prompt)
                expert = _request(
                    client,
                    endpoint=endpoint,
                    payload=_expert_payload(
                        model=args.model,
                        messages=_expert_phase_messages(manager_messages),
                        seed=_selector_seed(f"{task.data.name}:manager-expert", args.seed),
                    ),
                    tool_name="select_expert",
                    field="expert_id",
                )
                expert_rows.append(
                    {
                        "task": f"{task.data.name}:specialist-manager",
                        "family": family,
                        "role_scope": "nonroot_specialist_manager",
                        "expected": task.data.preferred_expert,
                        **expert,
                        "correct": expert["selected"] == task.data.preferred_expert,
                    }
                )

    action_correct = sum(row["correct"] for row in action_rows)
    action_classes = {row["expected"] for row in action_rows}
    correct_action_classes = {row["expected"] for row in action_rows if row["correct"]}
    expert_correct = Counter(
        row["expected"] for row in expert_rows if row["correct"]
    )
    manager_rows = [
        row for row in expert_rows if row["role_scope"] == "nonroot_specialist_manager"
    ]
    exact_transport = all(
        row["exact_one_field_typed"] for row in [*action_rows, *expert_rows]
    )
    acceptance = {
        "minimum_correct_root_actions": action_correct >= 12,
        "minimum_correct_per_root_action_class": action_classes <= correct_action_classes,
        "minimum_correct_generic_experts": expert_correct["generic_worker"] >= 1,
        "minimum_correct_table_experts": expert_correct["table_analyst"] >= 3,
        "minimum_correct_source_experts": expert_correct["source_inspector"] >= 3,
        "minimum_correct_recursive_manager_experts": sum(
            row["correct"] for row in manager_rows
        )
        >= 3,
        "exact_one_field_typed_transport": exact_transport,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "optimizer_updates": 0,
        "instance_offset": args.instance_offset,
        "seed": args.seed,
        "runtime_sources": runtime_sources,
        "root_action_correct": action_correct,
        "root_action_total": len(action_rows),
        "expert_correct_by_id": dict(sorted(expert_correct.items())),
        "expert_total_by_id": dict(
            sorted(Counter(row["expected"] for row in expert_rows).items())
        ),
        "recursive_manager_expert_correct": sum(row["correct"] for row in manager_rows),
        "recursive_manager_expert_total": len(manager_rows),
        "exact_one_field_typed_transport": exact_transport,
        "action_rows": action_rows,
        "expert_rows": expert_rows,
        "acceptance_gates_relaxed": False,
        "acceptance": acceptance,
        "split_routing_probe_passed": all(acceptance.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8101/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-traces", action="append", type=Path, required=True)
    parser.add_argument("--instance-offset", type=int, default=37300)
    parser.add_argument("--seed", type=int, default=20261205)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.instance_offset != 37300 or args.seed != 20261205:
        parser.error("split-routing probe offset and seed are frozen")
    args.runtime_traces = [path.resolve() for path in args.runtime_traces]
    result = probe(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite split-routing result: {args.output}")
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
