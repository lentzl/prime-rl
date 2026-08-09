#!/usr/bin/env python3
"""Export compact parent/child demonstrations for subagent communication rungs."""

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
        f"Read {path}, compute its weighted checksum, then send the integer checksum to "
        "your parent with agent_message before answering."
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


def _validate_no_repeated_tool_calls(examples: list[dict]) -> None:
    for example in examples:
        calls = [
            call["function"]["arguments"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        repeated = {call for call in calls if calls.count(call) > 1}
        if repeated:
            metadata = example["metadata"]
            raise ValueError(
                "repeated tool call in "
                f"{metadata['family']}:{metadata['task']}:{metadata['role']}"
            )


def _validate_followup_turn_boundaries(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "followup" or metadata["role"] != "parent":
            continue
        messages = example["messages"]
        incoming = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "user" and str(message.get("content", "")).startswith("[from child:")
        ]
        if len(incoming) != 2 or any(messages[index - 1]["role"] != "assistant" for index in incoming):
            raise ValueError(f"missing follow-up turn boundary in {metadata['task']}")


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
    local_assignment = f"local_values = {local_values!r}"
    local_checksum = (
        "local = sum((index + 1) * value for index, value in enumerate(local_values))\nlocal"
    )
    wait_code = (
        "import asyncio\n"
        "await asyncio.sleep(1)\n"
        "for _ in range(30):\n"
        "    child_state = await agent_observe.get_agent(handle.name)\n"
        "    if not child_state['agent']['isStreaming']:\n"
        "        break\n"
        "    await asyncio.sleep(0.5)\n"
        "child_state"
    )
    wait_output = (
        "{'agent': {'sessionName': 'shard-worker', 'status': 'idle', "
        "'isStreaming': False, 'messageCount': 5}}"
    )
    spawn_code = f"handle = await rlm({child_prompt!r}, name='shard-worker')"
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
            *_tool_messages("single-spawn", spawn_code, ""),
            *_tool_messages("single-local-assignment", local_assignment, ""),
            *_tool_messages("single-local-checksum", local_checksum, str(local)),
            *_tool_messages("single-wait", wait_code, wait_output),
            {"role": "user", "content": incoming},
            {"role": "assistant", "content": json.dumps(task.data.answer)},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "single", "task": task.data.name, "role": "parent"},
    }

    child_code = (
        f"import json\nfrom pathlib import Path\nvalues = json.loads(Path({path!r}).read_text())\n"
        "checksum = sum((index + 1) * value for index, value in enumerate(values))\n"
        "await agent_message.send(str(checksum), receiver_role='parent')"
    )
    send_output = (
        "{'id': 'agentmsg_training', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )
    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    child = {
        "messages": [
            {"role": "system", "content": child_system_prompt},
            {"role": "user", "content": f"[task from parent]\n\n{child_prompt}"},
            *_tool_messages("child-compute-and-reply", child_code, send_output),
            {"role": "assistant", "content": "Sent the checksum to the parent."},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "single", "task": task.data.name, "role": "child"},
    }
    child["metadata"]["values"] = remote_values
    return [parent, child]


def _parallel_examples(task, prompt: str) -> list[dict]:
    local_values = _local_values(task.data.prompt)
    local = task.data.answer["local"]
    handles = {"alpha-worker": "alpha", "beta-worker": "beta"}
    spawn_messages = []
    child_prompts = {}
    for name in task.data.expected_children:
        path = task.data.child_paths[name]
        child_prompt = _child_prompt(path)
        child_prompts[name] = child_prompt
        spawn_messages.extend(
            _tool_messages(
                f"parallel-spawn-{handles[name]}",
                f"{handles[name]} = await rlm({child_prompt!r}, name={name!r})",
                "",
            )
        )

    local_assignment = f"local_values = {local_values!r}"
    local_checksum = (
        "local = sum((index + 1) * value for index, value in enumerate(local_values))\nlocal"
    )
    wait_code = (
        "import asyncio\n"
        "await asyncio.sleep(1)\n"
        "for _ in range(30):\n"
        "    child_states = await asyncio.gather(\n"
        "        agent_observe.get_agent(alpha.name),\n"
        "        agent_observe.get_agent(beta.name),\n"
        "    )\n"
        "    if all(not state['agent']['isStreaming'] for state in child_states):\n"
        "        break\n"
        "    await asyncio.sleep(0.5)\n"
        "child_states"
    )
    wait_output = (
        "[{'agent': {'sessionName': 'alpha-worker', 'status': 'idle', "
        "'isStreaming': False, 'messageCount': 5}}, "
        "{'agent': {'sessionName': 'beta-worker', 'status': 'idle', "
        "'isStreaming': False, 'messageCount': 5}}]"
    )
    reply_order = list(task.data.expected_children)
    if task.data.template_variant % 2:
        reply_order.reverse()
    incoming = []
    for name in reply_order:
        key = handles[name]
        incoming.append(
            {
                "role": "user",
                "content": (
                    f"[from child:{name}]\n"
                    "Agent-to-agent message received.\n"
                    "Source: agent_message\n"
                    f"From: {name}\n"
                    f"Message id: agentmsg_training_{key}\n\n"
                    f"{task.data.answer[key]}"
                ),
            }
        )
    parent = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task.data.prompt},
            *spawn_messages,
            *_tool_messages("parallel-local-assignment", local_assignment, ""),
            *_tool_messages("parallel-local-checksum", local_checksum, str(local)),
            *_tool_messages("parallel-wait", wait_code, wait_output),
            *incoming,
            {"role": "assistant", "content": json.dumps(task.data.answer)},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "parallel", "task": task.data.name, "role": "parent"},
    }

    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    children = []
    for name in task.data.expected_children:
        key = handles[name]
        path = task.data.child_paths[name]
        values = json.loads(task.data.files[path])
        checksum = task.data.answer[key]
        compute_code = (
            f"import json\nfrom pathlib import Path\nvalues = json.loads(Path({path!r}).read_text())\n"
            "checksum = sum((index + 1) * value for index, value in enumerate(values))\n"
            "checksum"
        )
        send_code = "await agent_message.send(str(checksum), receiver_role='parent')"
        send_output = (
            f"{{'id': 'agentmsg_training_{key}', 'source': 'agent_message', "
            "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
        )
        child = {
            "messages": [
                {"role": "system", "content": child_system_prompt},
                {"role": "user", "content": f"[task from parent]\n\n{child_prompts[name]}"},
                *_tool_messages(f"parallel-child-compute-{key}", compute_code, str(checksum)),
                *_tool_messages(f"parallel-child-send-{key}", send_code, send_output),
                {"role": "assistant", "content": "Sent the checksum to the parent."},
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "parallel",
                "task": task.data.name,
                "role": "child",
                "child": name,
                "values": values,
            },
        }
        children.append(child)
    return [parent, *children]


def _followup_examples(task, prompt: str) -> list[dict]:
    path = task.data.child_paths["key-worker"]
    values = json.loads(task.data.files[path])
    subtotal = task.data.answer["subtotal"]
    multiplier = task.data.answer["multiplier"]
    result = task.data.answer["result"]
    child_prompt = (
        f"You are key-worker, my child. Read {path} and retain its subtotal. Do not call rlm or "
        "message a child. In a separate IPython call, send 'need multiplier' exactly with "
        "receiver_role='parent'. End that turn and resume only when my parent follow-up arrives. "
        "Then multiply the retained subtotal and send a JSON object containing subtotal and "
        "result with receiver_role='parent' before answering."
    )
    spawn_code = f"child = await rlm({child_prompt!r}, name='key-worker')"
    request_message = {
        "role": "user",
        "content": (
            "[from child:key-worker]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            "From: key-worker\n"
            "Message id: agentmsg_training_request\n\n"
            "need multiplier"
        ),
    }
    reply_code = (
        f"multiplier = {multiplier}\n"
        "await agent_message.send(str(multiplier), receiver_role='child', "
        "receiver_name=child.name)"
    )
    reply_output = (
        "{'id': 'agentmsg_training_multiplier', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'child'}"
    )
    final_payload = json.dumps({"subtotal": subtotal, "result": result})
    final_message = {
        "role": "user",
        "content": (
            "[from child:key-worker]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            "From: key-worker\n"
            "Message id: agentmsg_training_result\n\n"
            f"{final_payload}"
        ),
    }
    parent = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task.data.prompt},
            *_tool_messages("followup-spawn", spawn_code, ""),
            {"role": "assistant", "content": "Spawned key-worker and retained its handle. Waiting for its request."},
            request_message,
            *_tool_messages("followup-send-multiplier", reply_code, reply_output),
            {"role": "assistant", "content": "Sent the multiplier to key-worker. Waiting for its result."},
            final_message,
            {"role": "assistant", "content": json.dumps(task.data.answer)},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "followup", "task": task.data.name, "role": "parent"},
    }

    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    compute_code = (
        f"import json\nfrom pathlib import Path\nvalues = json.loads(Path({path!r}).read_text())\n"
        "subtotal = sum(values)\n"
        "subtotal"
    )
    request_code = "await agent_message.send('need multiplier', receiver_role='parent')"
    request_output = (
        "{'id': 'agentmsg_training_request', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )
    parent_message = {
        "role": "user",
        "content": (
            "[from parent]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            "Message id: agentmsg_training_multiplier\n\n"
            f"{multiplier}"
        ),
    }
    result_code = (
        f"multiplier = {multiplier}\n"
        "result = subtotal * multiplier\n"
        "payload = json.dumps({'subtotal': subtotal, 'result': result})\n"
        "await agent_message.send(payload, receiver_role='parent')"
    )
    result_output = (
        "{'id': 'agentmsg_training_result', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )
    child = {
        "messages": [
            {"role": "system", "content": child_system_prompt},
            {"role": "user", "content": f"[task from parent]\n\n{child_prompt}"},
            *_tool_messages("followup-child-subtotal", compute_code, str(subtotal)),
            *_tool_messages("followup-child-request", request_code, request_output),
            {"role": "assistant", "content": "Requested the multiplier from the parent."},
            parent_message,
            *_tool_messages("followup-child-result", result_code, result_output),
            {"role": "assistant", "content": "Sent the final result to the parent."},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {
            "family": "followup",
            "task": task.data.name,
            "role": "child",
            "values": values,
        },
    }
    return [parent, child]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--instance-offset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--harness-trace", type=Path)
    parser.add_argument("--followup-copies", type=int, default=1)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("direct", "single", "parallel", "followup"),
        default=("direct", "single"),
    )
    args = parser.parse_args()
    if args.followup_copies < 1:
        parser.error("--followup-copies must be at least 1")

    prompt = system_prompt(args.harness_trace, SYSTEM_PROMPT)
    tasks = SubagentCommunicationTaskset(
        SubagentCommunicationConfig(
            split="train",
            families=tuple(args.families),
            instruction_level="standard",
            instances_per_template=args.instances,
            instance_offset=args.instance_offset,
            seed=args.seed,
        )
    ).load()
    examples = []
    for task in tasks:
        if task.data.family == "direct":
            examples.append(_direct_example(task, prompt))
        elif task.data.family == "single":
            examples.extend(_single_examples(task, prompt))
        elif task.data.family == "parallel":
            examples.extend(_parallel_examples(task, prompt))
        else:
            for copy in range(args.followup_copies):
                copied = _followup_examples(task, prompt)
                for example in copied:
                    example["metadata"]["copy"] = copy
                examples.extend(copied)

    _validate_no_repeated_tool_calls(examples)
    _validate_followup_turn_boundaries(examples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(example) + "\n" for example in examples))
    print(f"wrote {len(examples)} examples for {','.join(args.families)} to {args.output}")


if __name__ == "__main__":
    main()
