#!/usr/bin/env python3
"""Export full-document repair and grounded-claim trajectories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ipython_foundations_v1.document_control import (
    generate_document_control_scenario,
)
from ipython_foundations_v1.generators import TRAIN_VARIANTS
from ipython_foundations_v1.taskset import SYSTEM_PROMPT
from ipython_sft_export_utils import (
    IPYTHON_TOOL,
    replay_examples,
    system_prompt,
    tool_call,
)


def _example(variant: int, instance: int, prompt: str) -> dict:
    rng = random.Random((20260806 * 1_000_003) + (variant * 10_007) + instance)
    scenario = generate_document_control_scenario(variant, instance, rng)
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Work in the persistent IPython session. Treat the live traceback as "
                "feedback, prove that the correction succeeds, and preserve literal "
                f"source meaning.\n\n{scenario.instruction}"
            ),
        },
    ]
    for index, call in enumerate(scenario.expert_calls):
        call_id = f"document-control-{variant}-{instance}-{index}"
        messages.extend(
            (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call(call_id, call.code)],
                },
                {"role": "tool", "content": call.output, "tool_call_id": call_id},
            )
        )
    messages.append({"role": "assistant", "content": json.dumps(scenario.answer)})
    return {
        "messages": messages,
        "tools": [IPYTHON_TOOL],
        "metadata": {
            "family": "document_control",
            "variant": variant,
            "instance": instance,
            "file_kind": "pdf",
            "failure_kind": scenario.failure_kind,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--instances", type=int, default=6)
    parser.add_argument("--harness-trace", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()

    prompt = system_prompt(args.harness_trace, SYSTEM_PROMPT)
    document_examples = [
        _example(variant, instance, prompt) for instance in range(args.instances) for variant in TRAIN_VARIANTS
    ]
    replay = replay_examples(args.replay, {"assignment", "state", "file_processing"})
    examples = [*document_examples, *replay]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in examples))
    print(f"wrote {len(document_examples)} document-control and {len(replay)} replay examples to {args.output}")


if __name__ == "__main__":
    main()
