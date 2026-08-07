#!/usr/bin/env python3
"""Export supervised two-call trajectories for the IPython assignment rung."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ipython_foundations_v1.generators import TRAIN_VARIANTS, generate
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


def _example(
    variant: int,
    instance: int,
    round_index: int,
    system_prompt: str,
) -> dict:
    generated = generate("assignment", variant, instance, seed=20260806)
    round_ = generated.rounds[round_index]
    path, payload = next(iter(round_.files.items()))
    values = json.loads(payload)
    load_code = f"import json\nfrom pathlib import Path\nvalues = json.loads(Path({path!r}).read_text())"
    checksum_code = "sum((i + 1) * value for i, value in enumerate(values))"
    user_prompt = (
        "This is one continuing notebook session. Complete the current request, "
        "then retain useful IPython state for later requests.\n\n"
        f"{round_.instruction}"
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("assignment-load", load_code)],
            },
            {
                "role": "tool",
                "content": "",
                "tool_call_id": "assignment-load",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("assignment-use", checksum_code)],
            },
            {
                "role": "tool",
                "content": str(round_.answer),
                "tool_call_id": "assignment-use",
            },
            {"role": "assistant", "content": json.dumps(round_.answer)},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {
            "family": "assignment",
            "variant": variant,
            "instance": instance,
            "round": round_index,
            "values": values,
        },
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--harness-trace", type=Path)
    args = parser.parse_args()

    system_prompt = _system_prompt(args.harness_trace)
    examples = [
        _example(variant, instance, round_index, system_prompt)
        for instance in range(args.instances)
        for variant in TRAIN_VARIANTS
        for round_index in range(3)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in examples))
    print(f"wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
