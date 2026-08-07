#!/usr/bin/env python3
"""Export short file-processing trajectories with earlier notebook replay."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ipython_foundations_v1.file_processing import generate_file_processing_scenario
from ipython_foundations_v1.generators import TRAIN_VARIANTS
from ipython_foundations_v1.taskset import SYSTEM_PROMPT

IPYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "ipython",
        "description": (
            "Execute Python scratchpad code in a persistent IPython kernel. "
            "Variables, imports, and loaded data persist across calls."
        ),
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {"code": {"type": "string"}},
        },
    },
}


def _tool_call(call_id: str, code: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "ipython", "arguments": json.dumps({"code": code})},
    }


def _system_prompt(trace_path: Path | None) -> str:
    if trace_path is None:
        return SYSTEM_PROMPT
    for line in trace_path.read_text().splitlines():
        for trace in json.loads(line)["traces"]:
            for node in trace["nodes"]:
                message = node["message"]
                if message["role"] == "system" and isinstance(message.get("content"), str):
                    return message["content"]
    raise ValueError(f"no system message found in {trace_path}")


def _example(variant: int, instance: int, system_prompt: str) -> dict:
    rng = random.Random((20260806 * 1_000_003) + (variant * 10_007) + instance)
    scenario = generate_file_processing_scenario(variant, instance, rng)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Work in the persistent IPython session. Inspect live values and errors, "
                "preserve successful state, and change failed operations rather than "
                f"repeating them.\n\n{scenario.instruction}"
            ),
        },
    ]
    for index, call in enumerate(scenario.expert_calls):
        call_id = f"file-{variant}-{instance}-{index}"
        messages.extend(
            (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call(call_id, call.code)],
                },
                {"role": "tool", "content": call.output, "tool_call_id": call_id},
            )
        )
    messages.append({"role": "assistant", "content": json.dumps(scenario.answer)})
    return {
        "messages": messages,
        "tools": [IPYTHON_TOOL],
        "metadata": {
            "family": "file_processing",
            "variant": variant,
            "instance": instance,
            "file_kind": scenario.file_kind,
            "failure_kind": scenario.failure_kind,
            "terminal_status": scenario.terminal_status,
        },
    }


def _replay(path: Path | None) -> list[dict]:
    if path is None:
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    allowed = {"assignment", "state"}
    unexpected = {row.get("metadata", {}).get("family") for row in rows} - allowed
    if unexpected:
        raise ValueError(f"unsupported replay families in {path}: {sorted(unexpected)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--harness-trace", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()

    system_prompt = _system_prompt(args.harness_trace)
    file_examples = [
        _example(variant, instance, system_prompt) for instance in range(args.instances) for variant in TRAIN_VARIANTS
    ]
    replay_examples = _replay(args.replay)
    examples = [*file_examples, *replay_examples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in examples))
    print(f"wrote {len(file_examples)} file-processing and {len(replay_examples)} replay examples to {args.output}")


if __name__ == "__main__":
    main()
