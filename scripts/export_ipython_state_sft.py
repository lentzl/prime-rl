#!/usr/bin/env python3
"""Export supervised cross-request trajectories for the IPython state rung."""

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


def _user_prompt(instruction: str, previous_correct: bool | None) -> str:
    if previous_correct is None:
        prefix = (
            "This is one continuing notebook session. Complete the current request, "
            "then retain useful IPython state for later requests."
        )
    else:
        verdict = "passed" if previous_correct else "failed"
        prefix = (
            f"The previous answer {verdict} validation. The expected value is not "
            "revealed; continue from the existing notebook state."
        )
    return f"{prefix}\n\n{instruction}"


def _example(variant: int, instance: int, system_prompt: str) -> dict:
    generated = generate("state", variant, instance, seed=20260806)
    first, second, third = generated.rounds
    path, payload = next(iter(first.files.items()))
    records = json.loads(payload)

    load_code = f"import json\nfrom pathlib import Path\nrecords = json.loads(Path({path!r}).read_text())\nlen(records)"
    totals_code = (
        "totals = {}\n"
        "for row in records:\n"
        "    totals[row['group']] = totals.get(row['group'], 0) + row['amount']\n"
        "totals"
    )
    winners_code = (
        "largest = max(totals.values())\nsorted(label for label, total in totals.items() if total == largest)"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for index, (round_, code, output) in enumerate(
        (
            (first, load_code, str(first.answer)),
            (second, totals_code, repr(second.answer)),
            (third, winners_code, repr(third.answer)),
        )
    ):
        call_id = f"state-request-{index + 1}"
        messages.extend(
            (
                {
                    "role": "user",
                    "content": _user_prompt(round_.instruction, None if index == 0 else True),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call(call_id, code)],
                },
                {"role": "tool", "content": output, "tool_call_id": call_id},
                {"role": "assistant", "content": json.dumps(round_.answer)},
            )
        )

    return {
        "messages": messages,
        "tools": [IPYTHON_TOOL],
        "metadata": {
            "family": "state",
            "variant": variant,
            "instance": instance,
            "records": records,
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


def _assignment_replay(path: Path | None) -> list[dict]:
    if path is None:
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if any(row.get("metadata", {}).get("family") != "assignment" for row in rows):
        raise ValueError(f"non-assignment example found in replay data {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--harness-trace", type=Path)
    parser.add_argument("--assignment-replay", type=Path)
    args = parser.parse_args()

    system_prompt = _system_prompt(args.harness_trace)
    state_examples = [
        _example(variant, instance, system_prompt) for instance in range(args.instances) for variant in TRAIN_VARIANTS
    ]
    replay_examples = _assignment_replay(args.assignment_replay)
    examples = [*state_examples, *replay_examples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in examples))
    print(f"wrote {len(state_examples)} state and {len(replay_examples)} assignment examples to {args.output}")


if __name__ == "__main__":
    main()
