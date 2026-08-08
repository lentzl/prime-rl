#!/usr/bin/env python3
"""Export compact parent/child demonstrations for the first subagent rung."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ipython_sft_export_utils import IPYTHON_TOOL, system_prompt, tool_call
from subagent_communication_v1.taskset import (
    SYSTEM_PROMPT,
    SubagentCommunicationConfig,
    SubagentCommunicationTaskset,
)


def _local_values(prompt: str) -> list[int]:
    matched = re.search(r"(?:checksum of|local checksum for) (\[[^\]]+\])", prompt)
    if not matched:
        raise ValueError(f"could not find local values in prompt: {prompt}")
    values = json.loads(matched.group(1))
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"invalid local values in prompt: {matched.group(1)}")
    return values


def _child_prompt(path: str) -> str:
    return (
        f"Read {path}, compute its weighted checksum, then make a separate final IPython "
        "call exactly await agent_message.send(str(checksum), receiver_role='parent'). "
        "Your task is incomplete until that call succeeds; do not finish with only a normal "
        "assistant answer."
    )


def _tool_messages(call_id: str, code: str, output: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call(call_id, code)],
        },
        {"role": "tool", "content": output, "tool_call_id": call_id},
    ]


def _direct_example(task, prompt: str) -> dict:
    values = _local_values(task.data.prompt)
    code = (
        f"values = {values!r}\n"
        "checksum = sum((index + 1) * value for index, value in enumerate(values))\n"
        "checksum"
    )
    return {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task.data.prompt},
            *_tool_messages("direct-checksum", code, str(task.data.answer["checksum"])),
            {"role": "assistant", "content": json.dumps(task.data.answer)},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "direct", "task": task.data.name, "role": "parent"},
    }


def _single_examples(task, prompt: str) -> list[dict]:
    path = task.data.child_paths["shard-worker"]
    remote_values = json.loads(task.data.files[path])
    local_values = _local_values(task.data.prompt)
    local = task.data.answer["local"]
    remote = task.data.answer["remote"]
    child_prompt = _child_prompt(path)
    local_code = (
        f"local_values = {local_values!r}\n"
        "local = sum((index + 1) * value for index, value in enumerate(local_values))\n"
        "local"
    )
    spawn_code = f"handle = await rlm({child_prompt!r}, name='shard-worker')"
    spawn_output = (
        "RLMSpawnHandle(rlm_child_id='sub-training', name='shard-worker', "
        "session_dir=PosixPath('/tmp/prime-agent/sub-training'), model='current-policy')"
    )
    incoming = (
        "[from child:shard-worker]\n"
        "Agent-to-agent message received.\n"
        "Source: agent_message\n"
        "From: shard-worker\n"
        "Message id: agentmsg_training\n\n"
        f"{remote}"
    )
    parent = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task.data.prompt},
            *_tool_messages("single-local", local_code, str(local)),
            *_tool_messages("single-spawn", spawn_code, spawn_output),
            {"role": "user", "content": incoming},
            {"role": "assistant", "content": json.dumps(task.data.answer)},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "single", "task": task.data.name, "role": "parent"},
    }

    read_code = f"import json\nfrom pathlib import Path\nvalues = json.loads(Path({path!r}).read_text())"
    checksum_code = (
        "checksum = sum((index + 1) * value for index, value in enumerate(values))\n"
        "checksum"
    )
    send_code = "await agent_message.send(str(checksum), receiver_role='parent')"
    send_output = (
        "{'id': 'agentmsg_training', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )
    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    child = {
        "messages": [
            {"role": "system", "content": child_system_prompt},
            {"role": "user", "content": f"[task from parent]\n\n{child_prompt}"},
            *_tool_messages("child-read", read_code, ""),
            *_tool_messages("child-checksum", checksum_code, str(remote)),
            *_tool_messages("child-reply", send_code, send_output),
            {"role": "assistant", "content": "Sent the checksum to the parent."},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "single", "task": task.data.name, "role": "child"},
    }
    child["metadata"]["values"] = remote_values
    return [parent, child]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--harness-trace", type=Path)
    args = parser.parse_args()

    prompt = system_prompt(args.harness_trace, SYSTEM_PROMPT)
    tasks = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=("direct", "single"),
            instruction_level="standard",
            instances_per_template=args.instances,
        )
    ).load()
    examples = []
    for task in tasks:
        if task.data.family == "direct":
            examples.append(_direct_example(task, prompt))
        else:
            examples.extend(_single_examples(task, prompt))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(example) + "\n" for example in examples))
    print(f"wrote {len(examples)} direct/parent/child examples to {args.output}")


if __name__ == "__main__":
    main()
