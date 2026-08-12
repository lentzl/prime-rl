#!/usr/bin/env python3
"""Export compact parent/child demonstrations for subagent communication rungs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from ipython_sft_export_utils import IPYTHON_TOOL, system_prompt, tool_call
from subagent_admission_v1.taskset import SubagentAdmissionConfig, SubagentAdmissionTaskset
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
    formula = "sum((index + 1) * value for index, value in enumerate(values))"
    return (
        f"Read {path} as the JSON list values and compute its weighted checksum using exactly "
        f"{formula}. Then execute exactly once: "
        "await agent_message.send(str(checksum), receiver_role='parent'). After the delivery "
        "receipt succeeds, answer with a concise confirmation and stop."
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


def _call_names(code: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            names.add(f"{node.func.value.id}.{node.func.attr}")
    return names


def _current_system_prompt(trace_path: Path | None) -> str:
    prompt = system_prompt(trace_path, SYSTEM_PROMPT)
    marker = "Coordinate work through Prime Agent's persistent IPython kernel."
    prefix, separator, _ = prompt.rpartition(marker)
    if separator:
        return f"{prefix}{SYSTEM_PROMPT}"
    return f"{prompt.rstrip()}\n\n{SYSTEM_PROMPT}"


def _validate_no_repeated_tool_calls(examples: list[dict]) -> None:
    for example in examples:
        calls = [
            call["function"]["arguments"] for message in example["messages"] for call in message.get("tool_calls", [])
        ]
        repeated = {call for call in calls if calls.count(call) > 1}
        if repeated:
            metadata = example["metadata"]
            raise ValueError(f"repeated tool call in {metadata['family']}:{metadata['task']}:{metadata['role']}")


def _validate_event_driven_parent_turns(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["role"] != "parent":
            continue
        messages = example["messages"]
        incoming = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "user" and str(message.get("content", "")).startswith("[from child:")
        ]
        if any(messages[index - 1]["role"] != "assistant" for index in incoming):
            raise ValueError(f"missing incoming-message turn boundary in {metadata['task']}")
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in messages
            for call in message.get("tool_calls", [])
        ]
        if any("agent_observe" in code for code in calls):
            raise ValueError(f"parent polls instead of yielding in {metadata['task']}")


def _validate_followup_message_binding(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "followup" or metadata["role"] != "child":
            continue
        messages = example["messages"]
        parent_index = next(
            index
            for index, message in enumerate(messages)
            if message["role"] == "user" and str(message.get("content", "")).startswith("[from parent]")
        )
        body = str(messages[parent_index]["content"]).rsplit("\n\n", 1)[-1]
        result_call = next(call for message in messages[parent_index + 1 :] for call in message.get("tool_calls", []))
        code = json.loads(result_call["function"]["arguments"])["code"]
        expected = f"parent_message_body = {body!r}\nmultiplier = int(parent_message_body.strip())"
        if expected not in code or re.search(r"(?m)^multiplier\s*=\s*-?\d+\s*$", code):
            raise ValueError(f"follow-up child bypasses parent-message binding in {metadata['task']}")


def _validate_handshake_message_binding(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "handshake" or metadata["role"] != "child":
            continue
        messages = example["messages"]
        parent_index = next(
            index
            for index, message in enumerate(messages)
            if message["role"] == "user" and str(message.get("content", "")).startswith("[from parent]")
        )
        body = str(messages[parent_index]["content"]).rsplit("\n\n", 1)[-1]
        result_call = next(call for message in messages[parent_index + 1 :] for call in message.get("tool_calls", []))
        code = json.loads(result_call["function"]["arguments"])["code"]
        expected = f"parent_message_body = {body!r}\nnonce = int(parent_message_body.strip())"
        if expected not in code or re.search(r"(?m)^nonce\s*=\s*\d+\s*$", code):
            raise ValueError(f"handshake child bypasses parent-message binding in {metadata['task']}")


def _validate_parent_child_handle_continuity(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["role"] != "parent":
            continue
        messages = example["messages"]
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in messages
            for call in message.get("tool_calls", [])
        ]
        child_send_index = next(
            (index for index, code in enumerate(calls) if "receiver_name=child.name" in code),
            None,
        )
        if child_send_index is None:
            continue
        retained_in_trace = any("child = await rlm(" in code for code in calls[:child_send_index])
        resumed_with_handle = any("active as variable child" in str(message.get("content", "")) for message in messages)
        if not retained_in_trace and not resumed_with_handle:
            raise ValueError(f"child handle is unavailable in {metadata['task']}")


def _validate_retained_spawn_atoms(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "retained_spawn_atom":
            continue
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        if len(calls) != 1:
            raise ValueError(f"retained spawn atom has {len(calls)} calls in {metadata['task']}")
        code = calls[0]
        if not re.search(r"(?m)^multiplier = -?\d+$", code):
            raise ValueError(f"retained spawn atom omits local state in {metadata['task']}")
        if "child = await rlm(" not in code:
            raise ValueError(f"retained spawn atom discards child handle in {metadata['task']}")
        if "agent_message" in code or "agent_observe" in code:
            raise ValueError(f"retained spawn atom includes post-spawn work in {metadata['task']}")


def _add_retained_spawn_thinking(example: dict) -> None:
    assistant_messages = [
        message for message in example["messages"] if message["role"] == "assistant"
    ]
    if len(assistant_messages) != 2:
        raise ValueError(
            f"retained spawn atom has {len(assistant_messages)} assistant responses in "
            f"{example['metadata']['task']}"
        )
    assistant_messages[0]["reasoning_content"] = (
        "The multiplier is coordinator-owned state, while the file is child-owned work. "
        "I should retain the multiplier and child handle, give the child only its path and "
        "message contract, and not inspect the delegated file myself."
    )
    assistant_messages[1]["reasoning_content"] = (
        "The spawn succeeded. I should preserve the multiplier and child handle, end this "
        "turn, and wait for the child's explicit message instead of polling or observing it."
    )
    example["metadata"]["thinking_rationale"] = True


def _validate_retained_spawn_thinking(examples: list[dict]) -> None:
    for example in examples:
        if not example["metadata"].get("thinking_rationale"):
            continue
        reasoning = [
            str(message.get("reasoning_content", ""))
            for message in example["messages"]
            if message["role"] == "assistant"
        ]
        if len(reasoning) != 2 or not all(reasoning):
            raise ValueError(f"missing thinking rationale in {example['metadata']['task']}")
        combined = " ".join(reasoning)
        required = ("coordinator-owned", "child-owned", "child handle", "instead of polling")
        if not all(fragment in combined for fragment in required):
            raise ValueError(f"incomplete thinking rationale in {example['metadata']['task']}")


def _validate_binding_spawn_atoms(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "binding_spawn_atom":
            continue
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        if not calls:
            raise ValueError(f"binding spawn atom has no calls in {metadata['task']}")
        expected = "child = await rlm(spawn_prompt, name=child_name)"
        if calls[-1] != expected:
            raise ValueError(f"binding spawn atom has an invalid target in {metadata['task']}")
        if any(re.search(r"(?m)^await rlm\(", code) for code in calls):
            raise ValueError(f"binding spawn atom discards the child handle in {metadata['task']}")
        if any("agent_message" in code or "agent_observe" in code for code in calls):
            raise ValueError(f"binding spawn atom includes post-spawn work in {metadata['task']}")


def _validate_parallel_control_atoms(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "parallel_control_atom":
            continue
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        style = metadata["context_style"]
        if style == "child_send":
            expected = "await agent_message.send(str(checksum), receiver_role='parent')"
            if metadata["role"] != "child" or calls != [expected]:
                raise ValueError(f"parallel child does not send exactly once in {metadata['task']}")
            continue
        if metadata["role"] != "parent":
            raise ValueError(f"parallel control role is invalid in {metadata['task']}")
        if any(
            forbidden in code
            for code in calls
            for forbidden in ("agent_message", "agent_observe", "asyncio", "sleep(", "rlm(")
        ):
            raise ValueError(f"parallel parent polls or messages in {metadata['task']}")
        answer = example["messages"][-1]
        if answer["role"] != "assistant":
            raise ValueError(f"parallel parent atom has no terminal action in {metadata['task']}")
        if style == "parent_wait_for_both":
            if calls or not str(answer["content"]).startswith("Waiting for"):
                raise ValueError(f"parallel parent does not yield in {metadata['task']}")
        elif style == "parent_local_compute":
            if len(calls) != 1 or "local = sum((index + 1) * value" not in calls[0]:
                raise ValueError(f"parallel parent does not compute local evidence in {metadata['task']}")
        elif style == "parent_bind_first_and_wait":
            key = metadata["bound_key"]
            if len(calls) != 1 or f"{key} = int({key}_message_body.strip())" not in calls[0]:
                raise ValueError(f"parallel parent does not bind first reply in {metadata['task']}")
            if not str(answer["content"]).startswith("Waiting for"):
                raise ValueError(f"parallel parent does not yield after binding in {metadata['task']}")
        elif style == "parent_bind_second_and_fan_in":
            key = metadata["bound_key"]
            if len(calls) != 1 or f"{key} = int({key}_message_body.strip())" not in calls[0]:
                raise ValueError(f"parallel parent does not bind final reply in {metadata['task']}")
            if "'total': local + alpha + beta" not in calls[0]:
                raise ValueError(f"parallel parent does not execute fan-in in {metadata['task']}")
            expected = json.dumps(metadata["answer"])
            if answer["content"] != expected:
                raise ValueError(f"parallel parent fan-in has the wrong answer in {metadata['task']}")
        else:
            raise ValueError(f"unknown parallel control style {style!r} in {metadata['task']}")


def _validate_single_control_atoms(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "single_control_atom":
            continue
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        style = metadata["context_style"]
        if style == "child_send":
            expected = "await agent_message.send(str(checksum), receiver_role='parent')"
            if metadata["role"] != "child" or calls != [expected]:
                raise ValueError(f"single child does not send exactly once in {metadata['task']}")
            if example["messages"][-1] != {
                "role": "assistant",
                "content": "Sent the checksum to the parent.",
            }:
                raise ValueError(f"single child does not stop after sending in {metadata['task']}")
            continue
        if metadata["role"] != "parent":
            raise ValueError(f"single control role is invalid in {metadata['task']}")
        call_names = {name for code in calls for name in _call_names(code)}
        if (
            "agent_message.send" in call_names
            or any(name.startswith("agent_observe.") for name in call_names)
            or "asyncio.sleep" in call_names
            or "time.sleep" in call_names
        ):
            raise ValueError(f"single parent polls or messages in {metadata['task']}")
        if style == "parent_spawn_exact_payload":
            if len(calls) != 1 or "handle = await rlm(" not in calls[0]:
                raise ValueError(f"single parent does not retain its spawn in {metadata['task']}")
            if metadata["path"] not in calls[0] or "name='shard-worker'" not in calls[0]:
                raise ValueError(f"single parent omits its exact payload in {metadata['task']}")
        elif style == "parent_local_compute":
            if len(calls) != 1 or "local = sum((index + 1) * value" not in calls[0]:
                raise ValueError(f"single parent does not compute local evidence in {metadata['task']}")
            if "rlm(" in calls[0]:
                raise ValueError(f"single parent respawns during local work in {metadata['task']}")
        elif style == "parent_wait_for_reply":
            if calls or not str(example["messages"][-1]["content"]).startswith("Waiting for"):
                raise ValueError(f"single parent does not yield in {metadata['task']}")
        elif style == "parent_bind_reply_and_finalize":
            if len(calls) != 1 or "remote = int(remote_message_body.strip())" not in calls[0]:
                raise ValueError(f"single parent does not bind its reply in {metadata['task']}")
            if "'total': local + remote" not in calls[0]:
                raise ValueError(f"single parent does not execute fan-in in {metadata['task']}")
            if example["messages"][-1]["content"] != json.dumps(metadata["answer"]):
                raise ValueError(f"single parent fan-in has the wrong answer in {metadata['task']}")
        else:
            raise ValueError(f"unknown single control style {style!r} in {metadata['task']}")


def _validate_single_recovery_atoms(examples: list[dict]) -> None:
    child_send = "await agent_message.send(str(checksum), receiver_role='parent')"
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "single_recovery_atom":
            continue
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        if len(calls) != 1:
            raise ValueError(f"single recovery must make one corrective call in {metadata['task']}")
        context = str(example["messages"][1]["content"])
        style = metadata["context_style"]
        if style == "child_string_type_repair":
            if metadata["role"] != "child" or calls != [child_send]:
                raise ValueError(f"single child does not repair its message type in {metadata['task']}")
            if "TypeError: message must be str" not in context:
                raise ValueError(f"single child repair omits the real type error in {metadata['task']}")
        elif style == "child_declared_tool_repair":
            if metadata["role"] != "child" or calls != [child_send]:
                raise ValueError(f"single child does not repair its tool route in {metadata['task']}")
            if "Tool agent_message not found" not in context:
                raise ValueError(f"single child repair omits the tool error in {metadata['task']}")
        elif style == "parent_top_level_await_repair":
            if metadata["role"] != "parent" or not calls[0].startswith("handle = await rlm("):
                raise ValueError(f"single parent does not use top-level await in {metadata['task']}")
            if metadata["path"] not in calls[0] or "RuntimeError" not in context:
                raise ValueError(f"single parent repair omits its path or traceback in {metadata['task']}")
            if any(fragment in calls[0] for fragment in ("asyncio", "run_until_complete", "local =")):
                raise ValueError(f"single parent repeats the failed operation in {metadata['task']}")
        else:
            raise ValueError(f"unknown single recovery style {style!r} in {metadata['task']}")


def _validate_single_resumption_atoms(examples: list[dict]) -> None:
    send_code = "await agent_message.send(str(checksum), receiver_role='parent')"
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "single_resumption_atom":
            continue
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        style = metadata["context_style"]
        if style == "parent_inert_wait_repair":
            if metadata["role"] != "parent" or calls:
                raise ValueError(f"single parent does not replace its inert cell in {metadata['task']}")
            if example["messages"][-1] != {
                "role": "assistant",
                "content": "Waiting for shard-worker's explicit reply.",
            }:
                raise ValueError(f"single parent does not yield plainly in {metadata['task']}")
        elif style == "parent_message_finalize_without_tool":
            if metadata["role"] != "parent" or calls:
                raise ValueError(f"single parent uses a tool after fan-in in {metadata['task']}")
            if example["messages"][-1]["content"] != json.dumps(metadata["answer"]):
                raise ValueError(f"single parent finalizes with the wrong answer in {metadata['task']}")
            if not any(
                message["role"] == "user" and str(message.get("content", "")).startswith("[from child:")
                for message in example["messages"]
            ):
                raise ValueError(f"single parent finalizes without a child event in {metadata['task']}")
        elif style == "child_inert_send_repair":
            if metadata["role"] != "child" or calls != [send_code]:
                raise ValueError(f"single child does not replace its inert cell in {metadata['task']}")
        elif style == "child_stop_after_receipt":
            if metadata["role"] != "child" or calls:
                raise ValueError(f"single child keeps using tools after delivery in {metadata['task']}")
            if example["messages"][-1] != {
                "role": "assistant",
                "content": "Sent the checksum to the parent.",
            }:
                raise ValueError(f"single child does not stop after delivery in {metadata['task']}")
        else:
            raise ValueError(f"unknown single resumption style {style!r} in {metadata['task']}")


def _validate_single_admission_trajectories(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "single_admission_trajectory":
            continue
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        path = metadata["path"]
        if metadata["role"] != "parent":
            raise ValueError(f"single admission is not a parent in {metadata['task']}")
        if len(calls) != 2:
            raise ValueError(f"single admission must contain two calls in {metadata['task']}")
        if not calls[0].startswith("handle = await rlm(") or path not in calls[0]:
            raise ValueError(f"single admission has an invalid spawn in {metadata['task']}")
        if "agent_message" not in calls[0] or "name='shard-worker'" not in calls[0]:
            raise ValueError(f"single admission omits its reply contract in {metadata['task']}")
        if path in calls[1] or "local = sum(" not in calls[1]:
            raise ValueError(f"single admission has invalid local work in {metadata['task']}")
        if any(
            message["role"] == "user" and message["content"].startswith("[from child:shard-worker]")
            for message in example["messages"]
        ):
            raise ValueError(f"single admission includes fan-in in {metadata['task']}")
        if example["messages"][-1] != {
            "role": "assistant",
            "content": "Waiting for shard-worker's explicit reply.",
        }:
            raise ValueError(f"single admission does not yield in {metadata['task']}")


def _validate_parallel_parent_bindings(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] != "parallel" or metadata["role"] != "parent":
            continue
        messages = example["messages"]
        for index, message in enumerate(messages):
            content = str(message.get("content", ""))
            matched = re.match(r"^\[from child:(alpha|beta)-worker\]", content)
            if not matched:
                continue
            key = matched.group(1)
            body = content.rsplit("\n\n", 1)[-1]
            call = messages[index + 1]["tool_calls"][0]
            code = json.loads(call["function"]["arguments"])["code"]
            expected = f"{key}_message_body = {body!r}\n{key} = int({key}_message_body.strip())\n{key}"
            if code != expected:
                raise ValueError(f"parallel parent does not bind {key} in {metadata['task']}")
        final_call = next(call for message in reversed(messages) for call in message.get("tool_calls", []))
        final_code = json.loads(final_call["function"]["arguments"])["code"]
        if "'total': local + alpha + beta" not in final_code:
            raise ValueError(f"parallel parent does not compute fan-in in {metadata['task']}")


def _validate_atomic_local_computation(examples: list[dict]) -> None:
    for example in examples:
        metadata = example["metadata"]
        if metadata["family"] not in {"single", "parallel"} or metadata["role"] != "parent":
            continue
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        local_calls = [code for code in calls if "local_values =" in code or "local = sum((index + 1) * value" in code]
        if len(local_calls) != 1:
            raise ValueError(f"local computation is split across cells in {metadata['task']}")
        if not all(fragment in local_calls[0] for fragment in ("local_values =", "local = sum((index + 1) * value")):
            raise ValueError(f"local computation is incomplete in {metadata['task']}")


def _direct_example(task, prompt: str) -> dict:
    values = _local_values(task.data.prompt)
    code = f"values = {values!r}\nchecksum = sum((index + 1) * value for index, value in enumerate(values))\nchecksum"
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
    local_compute = (
        f"local_values = {local_values!r}\n"
        "local = sum((index + 1) * value for index, value in enumerate(local_values))\n"
        "local"
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
            *_tool_messages("single-local-compute", local_compute, str(local)),
            {
                "role": "assistant",
                "content": "Waiting for shard-worker's explicit reply.",
            },
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


def _single_admission_example(task, prompt: str) -> dict:
    parent = _single_examples(task, prompt)[0]
    incoming_index = next(
        index
        for index, message in enumerate(parent["messages"])
        if message["role"] == "user" and message["content"].startswith("[from child:shard-worker]")
    )
    parent["messages"] = parent["messages"][:incoming_index]
    parent["metadata"] = {
        **parent["metadata"],
        "family": "single_admission_trajectory",
        "context_style": "exact_spawn_local_yield",
        "path": task.data.child_paths["shard-worker"],
    }
    return parent


def _single_recovery_atoms(task, prompt: str) -> list[dict]:
    path = task.data.child_paths["shard-worker"]
    remote = task.data.answer["remote"]
    child_prompt = _child_prompt(path)
    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    send_code = "await agent_message.send(str(checksum), receiver_role='parent')"
    send_output = (
        "{'id': 'agentmsg_single_recovery', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )
    child_common = {
        "tools": [IPYTHON_TOOL],
        "metadata": {
            "family": "single_recovery_atom",
            "task": task.data.name,
            "role": "child",
            "path": path,
        },
    }
    return [
        {
            "messages": [
                {"role": "system", "content": child_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"[task from parent]\n\n{child_prompt}\n\n"
                        f"The persistent kernel retains checksum = {remote}. The previous cell "
                        "failed without sending a message:\n"
                        "await agent_message.send(checksum, receiver_role='parent')\n"
                        "TypeError: message must be str, got int\n\n"
                        "Use the traceback to correct only the argument type, send once, and stop."
                    ),
                },
                *_tool_messages("single-recovery-string", send_code, send_output),
                {"role": "assistant", "content": "Sent the checksum to the parent."},
            ],
            **child_common,
            "metadata": {
                **child_common["metadata"],
                "context_style": "child_string_type_repair",
            },
        },
        {
            "messages": [
                {"role": "system", "content": child_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"[task from parent]\n\n{child_prompt}\n\n"
                        f"The persistent kernel retains checksum = {remote}. A direct tool call "
                        "failed without sending a message:\n"
                        "Tool agent_message not found. Available tools: ipython.\n\n"
                        "Route the installed Python module through the declared IPython tool, "
                        "send once, and stop."
                    ),
                },
                *_tool_messages("single-recovery-tool-route", send_code, send_output),
                {"role": "assistant", "content": "Sent the checksum to the parent."},
            ],
            **child_common,
            "metadata": {
                **child_common["metadata"],
                "context_style": "child_declared_tool_repair",
            },
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Resume this coordinator task:\n\n{task.data.prompt}\n\n"
                        "No child was admitted. The previous cell tried to create and run a new "
                        "event loop around rlm and failed:\n"
                        "RuntimeError: Cannot run the event loop while another loop is running\n\n"
                        "Prime Agent IPython supports top-level await. Correct only the spawn, "
                        "retain its handle, and do not compute local work in this cell."
                    ),
                },
                *_tool_messages(
                    "single-recovery-top-level-await",
                    f"handle = await rlm({child_prompt!r}, name='shard-worker')",
                    "",
                ),
                {
                    "role": "assistant",
                    "content": "Spawned shard-worker with the exact contract and retained its handle.",
                },
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "single_recovery_atom",
                "task": task.data.name,
                "role": "parent",
                "path": path,
                "context_style": "parent_top_level_await_repair",
            },
        },
    ]


def _single_resumption_atoms(task, prompt: str) -> list[dict]:
    path = task.data.child_paths["shard-worker"]
    local = task.data.answer["local"]
    remote = task.data.answer["remote"]
    child_prompt = _child_prompt(path)
    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    incoming = {
        "role": "user",
        "content": (
            "[from child:shard-worker]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            "From: shard-worker\n"
            "Message id: agentmsg_single_resumption\n\n"
            f"{remote}"
        ),
    }
    retained_parent = (
        f"Resume this coordinator task:\n\n{task.data.prompt}\n\n"
        "The persistent kernel retains the admitted child as handle and retains "
        f"local = {local}. Do not spawn again or inspect the delegated path. "
    )
    child_send = "await agent_message.send(str(checksum), receiver_role='parent')"
    child_receipt = (
        "{'id': 'agentmsg_single_resumption', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )
    return [
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"{retained_parent}No child message has arrived. The previous IPython "
                        "cell contained only `# Check for child messages` and returned no output. "
                        "That inert cell changed no state. Do not repeat or replace it with another "
                        "tool call; end this continuation with the required plain waiting response."
                    ),
                },
                {"role": "assistant", "content": "Waiting for shard-worker's explicit reply."},
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "single_resumption_atom",
                "task": task.data.name,
                "role": "parent",
                "path": path,
                "context_style": "parent_inert_wait_repair",
            },
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"{retained_parent}No child message has arrived yet. End the turn without "
                        "an IPython placeholder and preserve the retained state."
                    ),
                },
                {"role": "assistant", "content": "Waiting for shard-worker's explicit reply."},
                incoming,
                {"role": "assistant", "content": json.dumps(task.data.answer)},
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "single_resumption_atom",
                "task": task.data.name,
                "role": "parent",
                "path": path,
                "context_style": "parent_message_finalize_without_tool",
                "answer": task.data.answer,
            },
        },
        {
            "messages": [
                {"role": "system", "content": child_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"[task from parent]\n\n{child_prompt}\n\n"
                        f"The persistent kernel retains checksum = {remote}. The previous IPython "
                        "cell contained only `# Wait for parent` and returned no output. It changed "
                        "no state. Replace the inert action with the required send, execute it once, "
                        "and stop after the delivery receipt."
                    ),
                },
                *_tool_messages("single-resumption-child-send", child_send, child_receipt),
                {"role": "assistant", "content": "Sent the checksum to the parent."},
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "single_resumption_atom",
                "task": task.data.name,
                "role": "child",
                "path": path,
                "context_style": "child_inert_send_repair",
            },
        },
        {
            "messages": [
                {"role": "system", "content": child_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"[task from parent]\n\n{child_prompt}\n\n"
                        f"The persistent kernel retains checksum = {remote}. The required send "
                        "already succeeded and returned this delivery receipt:\n"
                        f"{child_receipt}\n\nMake no more tool calls. Confirm delivery concisely and stop."
                    ),
                },
                {"role": "assistant", "content": "Sent the checksum to the parent."},
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "single_resumption_atom",
                "task": task.data.name,
                "role": "child",
                "path": path,
                "context_style": "child_stop_after_receipt",
            },
        },
    ]


def _single_control_atoms(task, prompt: str) -> list[dict]:
    path = task.data.child_paths["shard-worker"]
    local_values = _local_values(task.data.prompt)
    local = task.data.answer["local"]
    remote = task.data.answer["remote"]
    child_prompt = _child_prompt(path)
    spawn_code = f"handle = await rlm({child_prompt!r}, name='shard-worker')"
    local_code = (
        f"local_values = {local_values!r}\n"
        "local = sum((index + 1) * value for index, value in enumerate(local_values))\n"
        "local"
    )
    retained = (
        "The persistent kernel retains handle for the one successfully spawned child named "
        "'shard-worker'. Do not spawn another child. "
    )
    incoming = {
        "role": "user",
        "content": (
            "[from child:shard-worker]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            "From: shard-worker\n"
            "Message id: agentmsg_single_control\n\n"
            f"{remote}"
        ),
    }
    fan_in_code = (
        f"remote_message_body = {str(remote)!r}\n"
        "remote = int(remote_message_body.strip())\n"
        "answer = {\n"
        "    'local': local,\n"
        "    'remote': remote,\n"
        "    'total': local + remote,\n"
        "}\n"
        "answer"
    )
    common = {
        "tools": [IPYTHON_TOOL],
        "metadata": {
            "family": "single_control_atom",
            "task": task.data.name,
            "role": "parent",
            "path": path,
        },
    }
    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    child_send_code = "await agent_message.send(str(checksum), receiver_role='parent')"
    child_send_output = (
        "{'id': 'agentmsg_single_control_child', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )
    return [
        {
            "messages": [
                {"role": "system", "content": child_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"[task from parent]\n\n{child_prompt}\n\n"
                        f"The successful prior IPython cell retained checksum = {remote} but did "
                        "not send it. Preserve that value, send it to the parent exactly once with "
                        "agent_message, then stop."
                    ),
                },
                *_tool_messages(
                    "single-control-child-send",
                    child_send_code,
                    child_send_output,
                ),
                {"role": "assistant", "content": "Sent the checksum to the parent."},
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "single_control_atom",
                "task": task.data.name,
                "role": "child",
                "path": path,
                "context_style": "child_send",
            },
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": task.data.prompt},
                *_tool_messages("single-control-spawn", spawn_code, ""),
                {
                    "role": "assistant",
                    "content": "Spawned shard-worker with the exact shard path and retained its handle.",
                },
            ],
            **common,
            "metadata": {
                **common["metadata"],
                "context_style": "parent_spawn_exact_payload",
            },
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Resume this coordinator task:\n\n{task.data.prompt}\n\n{retained}"
                        "The coordinator-local checksum has not been computed. Compute it in one "
                        "IPython cell, retain it as local, then end the turn without polling."
                    ),
                },
                *_tool_messages("single-control-local", local_code, str(local)),
                {"role": "assistant", "content": "Waiting for shard-worker's explicit reply."},
            ],
            **common,
            "metadata": {**common["metadata"], "context_style": "parent_local_compute"},
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Resume this coordinator task:\n\n{task.data.prompt}\n\n{retained}"
                        f"The persistent kernel also retains local = {local}. No child message "
                        "has arrived. End this turn without polling or repeating a successful cell."
                    ),
                },
                {"role": "assistant", "content": "Waiting for shard-worker's explicit reply."},
            ],
            **common,
            "metadata": {**common["metadata"], "context_style": "parent_wait_for_reply"},
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Resume this coordinator task:\n\n{task.data.prompt}\n\n{retained}"
                        f"The persistent kernel also retains local = {local}. Wait for the child "
                        "message, bind its body as remote, compute the final object, and stop."
                    ),
                },
                {"role": "assistant", "content": "Waiting for shard-worker's explicit reply."},
                incoming,
                *_tool_messages("single-control-fan-in", fan_in_code, repr(task.data.answer)),
                {"role": "assistant", "content": json.dumps(task.data.answer)},
            ],
            **common,
            "metadata": {
                **common["metadata"],
                "context_style": "parent_bind_reply_and_finalize",
                "answer": task.data.answer,
            },
        },
    ]


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

    local_compute = (
        f"local_values = {local_values!r}\n"
        "local = sum((index + 1) * value for index, value in enumerate(local_values))\n"
        "local"
    )
    reply_order = list(task.data.expected_children)
    if task.data.template_variant % 2:
        reply_order.reverse()
    incoming: list[tuple[str, str, int, dict]] = []
    for name in reply_order:
        key = handles[name]
        value = task.data.answer[key]
        incoming.append(
            (
                name,
                key,
                value,
                {
                    "role": "user",
                    "content": (
                        f"[from child:{name}]\n"
                        "Agent-to-agent message received.\n"
                        "Source: agent_message\n"
                        f"From: {name}\n"
                        f"Message id: agentmsg_training_{key}\n\n"
                        f"{value}"
                    ),
                },
            )
        )
    fan_in_messages: list[dict] = []
    for index, (name, key, value, message) in enumerate(incoming):
        fan_in_messages.append(message)
        binding_code = f"{key}_message_body = {str(value)!r}\n{key} = int({key}_message_body.strip())\n{key}"
        fan_in_messages.extend(_tool_messages(f"parallel-bind-{key}", binding_code, str(value)))
        if index < len(incoming) - 1:
            fan_in_messages.append(
                {
                    "role": "assistant",
                    "content": (f"Bound {name}'s reply as {key}. Waiting for the remaining child."),
                }
            )
    fan_in_code = (
        "answer = {\n"
        "    'local': local,\n"
        "    'alpha': alpha,\n"
        "    'beta': beta,\n"
        "    'total': local + alpha + beta,\n"
        "}\n"
        "answer"
    )
    parent = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task.data.prompt},
            *spawn_messages,
            *_tool_messages("parallel-local-compute", local_compute, str(local)),
            {
                "role": "assistant",
                "content": "Finished coordinator-local work. Waiting for both explicit child replies.",
            },
            *fan_in_messages,
            *_tool_messages(
                "parallel-semantic-fan-in",
                fan_in_code,
                repr(task.data.answer),
            ),
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


def _parallel_control_atoms(task, prompt: str) -> list[dict]:
    local = task.data.answer["local"]
    local_values = _local_values(task.data.prompt)
    children = list(task.data.expected_children)
    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    send_code = "await agent_message.send(str(checksum), receiver_role='parent')"
    atoms = []
    for name in children:
        key = "alpha" if name == "alpha-worker" else "beta"
        path = task.data.child_paths[name]
        checksum = task.data.answer[key]
        atoms.append(
            {
                "messages": [
                    {"role": "system", "content": child_system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"[task from parent]\n\n{_child_prompt(path)}\n\n"
                            f"The successful prior cell retained checksum = {checksum} but did not "
                            "send it. Preserve that value, send it to the parent exactly once with "
                            "agent_message, then stop."
                        ),
                    },
                    *_tool_messages(
                        f"parallel-control-child-send-{key}",
                        send_code,
                        (
                            f"{{'id': 'agentmsg_control_{key}', 'source': 'agent_message', "
                            "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
                        ),
                    ),
                    {"role": "assistant", "content": "Sent the checksum to the parent."},
                ],
                "tools": [IPYTHON_TOOL],
                "metadata": {
                    "family": "parallel_control_atom",
                    "task": task.data.name,
                    "role": "child",
                    "child": name,
                    "context_style": "child_send",
                },
            }
        )

    retained = (
        f"The coordinator's persistent kernel retains local = {local}, plus child handles "
        f"{children[0]!r} and {children[1]!r}. Both children were spawned successfully. "
    )
    local_code = (
        f"local_values = {local_values!r}\n"
        "local = sum((index + 1) * value for index, value in enumerate(local_values))\n"
        "local"
    )
    atoms.append(
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Resume this coordinator task:\n\n{task.data.prompt}\n\n"
                        f"Child handles {children[0]!r} and {children[1]!r} are already retained. "
                        "The coordinator-local checksum has not been computed. Execute the exact "
                        "weighted checksum in IPython, retain it as local, then end this turn "
                        "without polling."
                    ),
                },
                *_tool_messages("parallel-control-local", local_code, str(local)),
                {"role": "assistant", "content": "Waiting for both explicit child replies."},
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "parallel_control_atom",
                "task": task.data.name,
                "role": "parent",
                "context_style": "parent_local_compute",
            },
        }
    )
    atoms.append(
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Resume this coordinator task:\n\n{task.data.prompt}\n\n{retained}"
                        "No child message has arrived. End this turn without polling, messaging a "
                        "parent, or repeating a successful cell."
                    ),
                },
                {"role": "assistant", "content": "Waiting for both explicit child replies."},
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "parallel_control_atom",
                "task": task.data.name,
                "role": "parent",
                "context_style": "parent_wait_for_both",
            },
        }
    )

    reply_order = children.copy()
    if task.data.template_variant % 2:
        reply_order.reverse()
    first, remaining = reply_order
    first_key = "alpha" if first == "alpha-worker" else "beta"
    remaining_key = "alpha" if remaining == "alpha-worker" else "beta"
    first_value = task.data.answer[first_key]
    remaining_value = task.data.answer[remaining_key]
    first_binding_code = (
        f"{first_key}_message_body = {str(first_value)!r}\n"
        f"{first_key} = int({first_key}_message_body.strip())\n"
        f"{first_key}"
    )
    atoms.append(
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Resume this coordinator task:\n\n{task.data.prompt}\n\n{retained}"
                        "No child message has arrived yet."
                    ),
                },
                {"role": "assistant", "content": "Waiting for both explicit child replies."},
                {
                    "role": "user",
                    "content": (
                        f"[from child:{first}]\n"
                        "Agent-to-agent message received.\n"
                        "Source: agent_message\n"
                        f"From: {first}\n"
                        f"Message id: agentmsg_control_{first_key}\n\n"
                        f"{first_value}"
                    ),
                },
                *_tool_messages(
                    f"parallel-control-bind-{first_key}",
                    first_binding_code,
                    str(first_value),
                ),
                {
                    "role": "assistant",
                    "content": f"Waiting for {remaining}'s explicit reply.",
                },
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "parallel_control_atom",
                "task": task.data.name,
                "role": "parent",
                "context_style": "parent_bind_first_and_wait",
                "bound_key": first_key,
            },
        }
    )

    final_code = (
        f"{remaining_key}_message_body = {str(remaining_value)!r}\n"
        f"{remaining_key} = int({remaining_key}_message_body.strip())\n"
        "answer = {\n"
        "    'local': local,\n"
        "    'alpha': alpha,\n"
        "    'beta': beta,\n"
        "    'total': local + alpha + beta,\n"
        "}\n"
        "answer"
    )
    atoms.append(
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Resume this coordinator task:\n\n{task.data.prompt}\n\n"
                        f"The persistent kernel retains local = {local} and {first_key} = "
                        f"{first_value}, bound from {first}'s explicit message. "
                        f"Wait for {remaining}'s explicit reply."
                    ),
                },
                {"role": "assistant", "content": f"Waiting for {remaining}'s explicit reply."},
                {
                    "role": "user",
                    "content": (
                        f"[from child:{remaining}]\n"
                        "Agent-to-agent message received.\n"
                        "Source: agent_message\n"
                        f"From: {remaining}\n"
                        f"Message id: agentmsg_control_{remaining_key}\n\n"
                        f"{remaining_value}"
                    ),
                },
                *_tool_messages(
                    f"parallel-control-bind-final-{remaining_key}",
                    final_code,
                    repr(task.data.answer),
                ),
                {"role": "assistant", "content": json.dumps(task.data.answer)},
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {
                "family": "parallel_control_atom",
                "task": task.data.name,
                "role": "parent",
                "context_style": "parent_bind_second_and_fan_in",
                "bound_key": remaining_key,
                "answer": task.data.answer,
            },
        }
    )
    return atoms


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
    spawn_code = f"multiplier = {multiplier}\nchild = await rlm({child_prompt!r}, name='key-worker')"
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
    reply_code = "await agent_message.send(str(multiplier), receiver_role='child', receiver_name=child.name)"
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
        f"parent_message_body = {str(multiplier)!r}\n"
        "multiplier = int(parent_message_body.strip())\n"
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


def _followup_protocol_atoms(task, prompt: str) -> list[dict]:
    path = task.data.child_paths["key-worker"]
    subtotal = task.data.answer["subtotal"]
    multiplier = task.data.answer["multiplier"]
    child_prompt = (
        f"You are key-worker, my child. Read {path} and retain its subtotal. Do not call rlm or "
        "message a child. Send 'need multiplier' with receiver_role='parent', end that turn, "
        "then resume on the parent message and send subtotal and result to the parent."
    )
    retain_multiplier = f"multiplier = {multiplier}"
    spawn_code = f"{retain_multiplier}\nchild = await rlm({child_prompt!r}, name='key-worker')"
    send_retained_multiplier = (
        "await agent_message.send(str(multiplier), receiver_role='child', receiver_name=child.name)"
    )
    repair_multiplier = (
        f"multiplier = {multiplier}\n"
        "await agent_message.send(str(multiplier), receiver_role='child', "
        "receiver_name=child.name)"
    )
    request_message = {
        "role": "user",
        "content": (
            "[from child:key-worker]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            "From: key-worker\n"
            "Message id: agentmsg_atom_request\n\n"
            "need multiplier"
        ),
    }
    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    receipt_to_child = (
        "{'id': 'agentmsg_atom_multiplier', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'child'}"
    )
    receipt_to_parent = (
        "{'id': 'agentmsg_atom_request', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )

    return [
        {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": task.data.prompt},
                *_tool_messages("atom-spawn", spawn_code, ""),
                {
                    "role": "assistant",
                    "content": "Spawned key-worker and retained its handle. Waiting for its request.",
                },
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {"family": "protocol_atom", "task": task.data.name, "role": "parent"},
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": task.data.prompt},
                *_tool_messages("atom-retain-parent-state", spawn_code, ""),
                {
                    "role": "assistant",
                    "content": (
                        "Spawned key-worker and retained both its handle and the multiplier. "
                        "Waiting for its explicit request."
                    ),
                },
                request_message,
                *_tool_messages(
                    "atom-parent-followup",
                    send_retained_multiplier,
                    receipt_to_child,
                ),
                {
                    "role": "assistant",
                    "content": "Sent the multiplier once to the existing child. Waiting for its result.",
                },
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {"family": "protocol_atom", "task": task.data.name, "role": "parent"},
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Resume the coordinator task below. key-worker is active as variable child.\n\n"
                        f"{task.data.prompt}\n\n"
                        "Its latest message is 'need multiplier'. The previous cell failed without "
                        "changing state:\n"
                        "multiplier = int(await agent_message.send(multiplier, "
                        "receiver_role='child', receiver_name=child.name))\n"
                        "NameError: name 'multiplier' is not defined\n\n"
                        "Use the multiplier stated in the task and correct only the failed operation."
                    ),
                },
                *_tool_messages("atom-parent-repair", repair_multiplier, receipt_to_child),
                {
                    "role": "assistant",
                    "content": "Corrected the failed operation and sent the multiplier once.",
                },
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {"family": "protocol_atom", "task": task.data.name, "role": "parent"},
        },
        {
            "messages": [
                {"role": "system", "content": child_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"[task from parent]\n\n{child_prompt}\n\n"
                        f"The subtotal is already retained as subtotal = {subtotal}. The previous "
                        "cell failed without changing state:\n"
                        "await agent_message('need multiplier', receiver_role='parent')\n"
                        "TypeError: 'module' object is not callable\n\n"
                        "Correct the API call once. Do not call rlm at depth one."
                    ),
                },
                *_tool_messages(
                    "atom-child-repair",
                    "await agent_message.send('need multiplier', receiver_role='parent')",
                    receipt_to_parent,
                ),
                {
                    "role": "assistant",
                    "content": "Requested the multiplier from the parent. Waiting for its message.",
                },
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {"family": "protocol_atom", "task": task.data.name, "role": "child"},
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Resume an existing coordinator turn. key-worker has already been spawned "
                        "successfully, but no child message has arrived yet. Preserve the child and "
                        "end this continuation without polling or another tool call."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Waiting for key-worker's explicit message.",
                },
            ],
            "tools": [IPYTHON_TOOL],
            "metadata": {"family": "protocol_atom", "task": task.data.name, "role": "parent"},
        },
    ]


def _handshake_examples(task, prompt: str) -> list[dict]:
    nonce = task.data.answer["nonce"]
    child_prompt = (
        "You are relay-worker, my child. Do not call rlm or message a child. In a separate "
        "IPython call, send 'need nonce' exactly with receiver_role='parent'. End that turn and "
        "resume only when my parent follow-up arrives. Bind the nonce from that message, send it "
        "back with receiver_role='parent', then stop."
    )
    spawn_code = f"nonce = {nonce}\nchild = await rlm({child_prompt!r}, name='relay-worker')"
    request_message = {
        "role": "user",
        "content": (
            "[from child:relay-worker]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            "From: relay-worker\n"
            "Message id: agentmsg_training_nonce_request\n\n"
            "need nonce"
        ),
    }
    followup_code = "await agent_message.send(str(nonce), receiver_role='child', receiver_name=child.name)"
    followup_output = (
        "{'id': 'agentmsg_training_nonce_followup', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'child'}"
    )
    result_message = {
        "role": "user",
        "content": (
            "[from child:relay-worker]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            "From: relay-worker\n"
            "Message id: agentmsg_training_nonce_result\n\n"
            f"{nonce}"
        ),
    }
    parent = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task.data.prompt},
            *_tool_messages("handshake-spawn", spawn_code, ""),
            {"role": "assistant", "content": "Spawned relay-worker and retained its handle. Waiting for its request."},
            request_message,
            *_tool_messages("handshake-send-nonce", followup_code, followup_output),
            {"role": "assistant", "content": "Sent the nonce to relay-worker. Waiting for its echo."},
            result_message,
            {"role": "assistant", "content": json.dumps(task.data.answer)},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "handshake", "task": task.data.name, "role": "parent"},
    }

    child_system_prompt = prompt.replace("Recursive agent depth: 0", "Recursive agent depth: 1")
    request_code = "await agent_message.send('need nonce', receiver_role='parent')"
    request_output = (
        "{'id': 'agentmsg_training_nonce_request', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )
    parent_message = {
        "role": "user",
        "content": (
            "[from parent]\n"
            "Agent-to-agent message received.\n"
            "Source: agent_message\n"
            "Message id: agentmsg_training_nonce_followup\n\n"
            f"{nonce}"
        ),
    }
    result_code = (
        f"parent_message_body = {str(nonce)!r}\n"
        "nonce = int(parent_message_body.strip())\n"
        "await agent_message.send(str(nonce), receiver_role='parent')"
    )
    result_output = (
        "{'id': 'agentmsg_training_nonce_result', 'source': 'agent_message', "
        "'deliveryStatus': 'delivered', 'receiverRole': 'parent'}"
    )
    child = {
        "messages": [
            {"role": "system", "content": child_system_prompt},
            {"role": "user", "content": f"[task from parent]\n\n{child_prompt}"},
            *_tool_messages("handshake-child-request", request_code, request_output),
            {"role": "assistant", "content": "Requested the nonce from the parent."},
            parent_message,
            *_tool_messages("handshake-child-result", result_code, result_output),
            {"role": "assistant", "content": "Echoed the nonce to the parent."},
        ],
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "handshake", "task": task.data.name, "role": "child"},
    }
    return [parent, child]


def _bidirectional_request_atom(task, prompt: str) -> dict:
    if task.data.family == "followup":
        child = _followup_examples(task, prompt)[1]
        waiting_text = "Requested the multiplier from the parent."
    else:
        child = _handshake_examples(task, prompt)[1]
        waiting_text = "Requested the nonce from the parent."
    waiting_index = next(
        index
        for index, message in enumerate(child["messages"])
        if message.get("role") == "assistant" and message.get("content") == waiting_text
    )
    child["messages"] = child["messages"][: waiting_index + 1]
    child["metadata"] = {
        "family": "bidirectional_request_atom",
        "source_family": task.data.family,
        "task": task.data.name,
        "role": "child",
    }
    return child


def _binding_focused_atoms(task, prompt: str) -> list[dict]:
    path = task.data.child_paths["key-worker"]
    multiplier = task.data.answer["multiplier"]
    child_prompt = (
        f"You are key-worker, my child. Read {path} and retain its subtotal. Do not call rlm or "
        "message a child. Send 'need multiplier' with receiver_role='parent', end that turn, "
        "then resume on the parent message and send subtotal and result to the parent."
    )
    setup_code = f"multiplier = {multiplier}\nspawn_prompt = {child_prompt!r}\nchild_name = 'key-worker'"
    target_code = "child = await rlm(spawn_prompt, name=child_name)"
    retained_state = (
        "The previous persistent-IPython cell completed successfully with no printed output. "
        "It retained the task multiplier plus these exact variables:\n"
        f"spawn_prompt = {child_prompt!r}\n"
        "child_name = 'key-worker'\n\n"
        "Continue from that state. Spawn exactly one child and retain the admission handle in "
        "the variable child. Do not repeat the successful assignments or do post-spawn work yet."
    )
    common = {
        "tools": [IPYTHON_TOOL],
        "metadata": {"family": "binding_spawn_atom", "task": task.data.name, "role": "parent"},
    }
    return [
        {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"{task.data.prompt}\n\n{retained_state}"},
                *_tool_messages("binding-retained-state", target_code, ""),
            ],
            **common,
            "metadata": {**common["metadata"], "context_style": "retained_state"},
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"{task.data.prompt}\n\n{retained_state}\n\n"
                        "The proposed next cell below would lose the admission handle:\n"
                        "await rlm(spawn_prompt, name=child_name)\n\n"
                        "Correct that operation before executing it."
                    ),
                },
                *_tool_messages("binding-correct-bare-spawn", target_code, ""),
            ],
            **common,
            "metadata": {**common["metadata"], "context_style": "correct_bare_spawn"},
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"{task.data.prompt}\n\n{retained_state}\n\n"
                        "The previous cell returned a coroutine because it omitted await:\n"
                        "child = rlm(spawn_prompt, name=child_name)\n"
                        "RuntimeWarning: coroutine 'run' was never awaited\n\n"
                        "Correct only that operation and execute it once."
                    ),
                },
                *_tool_messages("binding-correct-missing-await", target_code, ""),
            ],
            **common,
            "metadata": {**common["metadata"], "context_style": "correct_missing_await"},
        },
        {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": task.data.prompt},
                *_tool_messages("binding-silent-setup", setup_code, ""),
                *_tool_messages("binding-after-silent-setup", target_code, ""),
            ],
            **common,
            "metadata": {**common["metadata"], "context_style": "silent_cell_boundary"},
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--instance-offset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--harness-trace", type=Path)
    parser.add_argument("--followup-copies", type=int, default=1)
    parser.add_argument("--handshake-copies", type=int, default=1)
    parser.add_argument("--bidirectional-request-atoms", action="store_true")
    parser.add_argument("--bidirectional-request-copies", type=int, default=1)
    parser.add_argument("--protocol-atoms", action="store_true")
    parser.add_argument("--retained-spawn-only", action="store_true")
    parser.add_argument("--thinking-rationales", action="store_true")
    parser.add_argument("--binding-focused-only", action="store_true")
    parser.add_argument("--binding-focused-atoms", action="store_true")
    parser.add_argument("--extra-binding-instances", type=int, default=0)
    parser.add_argument("--parallel-control-atoms", action="store_true")
    parser.add_argument("--extra-parallel-control-instances", type=int, default=0)
    parser.add_argument("--extra-parallel-parent-instances", type=int, default=0)
    parser.add_argument("--single-control-atoms", action="store_true")
    parser.add_argument("--single-causal-mastery", action="store_true")
    parser.add_argument("--single-executable-causal-mastery", action="store_true")
    parser.add_argument("--single-resumption-control-mastery", action="store_true")
    parser.add_argument("--single-exact-spawn-only", action="store_true")
    parser.add_argument("--single-admission-only", action="store_true")
    parser.add_argument("--self-contained-child-contract", action="store_true")
    parser.add_argument("--extra-single-control-instances", type=int, default=0)
    parser.add_argument("--extra-single-parent-instances", type=int, default=0)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("direct", "single", "parallel", "followup", "handshake"),
        default=("direct", "single"),
    )
    args = parser.parse_args()
    if args.followup_copies < 1:
        parser.error("--followup-copies must be at least 1")
    if args.handshake_copies < 1:
        parser.error("--handshake-copies must be at least 1")
    if args.bidirectional_request_copies < 1:
        parser.error("--bidirectional-request-copies must be at least 1")
    if args.bidirectional_request_atoms and not {"followup", "handshake"}.intersection(args.families):
        parser.error("--bidirectional-request-atoms requires followup or handshake")
    if args.retained_spawn_only and tuple(args.families) != ("followup",):
        parser.error("--retained-spawn-only requires --families followup")
    if args.thinking_rationales and not args.retained_spawn_only:
        parser.error("--thinking-rationales requires --retained-spawn-only")
    if args.binding_focused_only and tuple(args.families) != ("followup",):
        parser.error("--binding-focused-only requires --families followup")
    if args.binding_focused_atoms and "followup" not in args.families:
        parser.error("--binding-focused-atoms requires the followup family")
    if args.extra_binding_instances < 0:
        parser.error("--extra-binding-instances must be nonnegative")
    if args.extra_binding_instances and not args.binding_focused_atoms:
        parser.error("--extra-binding-instances requires --binding-focused-atoms")
    if args.parallel_control_atoms and "parallel" not in args.families:
        parser.error("--parallel-control-atoms requires the parallel family")
    if args.extra_parallel_control_instances < 0:
        parser.error("--extra-parallel-control-instances must be nonnegative")
    if args.extra_parallel_control_instances and not args.parallel_control_atoms:
        parser.error("--extra-parallel-control-instances requires --parallel-control-atoms")
    if args.extra_parallel_parent_instances < 0:
        parser.error("--extra-parallel-parent-instances must be nonnegative")
    if args.extra_parallel_parent_instances and "parallel" not in args.families:
        parser.error("--extra-parallel-parent-instances requires the parallel family")
    if args.single_control_atoms and "single" not in args.families:
        parser.error("--single-control-atoms requires the single family")
    if args.single_causal_mastery and tuple(args.families) not in {
        ("single",),
        ("direct", "single"),
    }:
        parser.error("--single-causal-mastery supports only direct and single families")
    if args.single_executable_causal_mastery and tuple(args.families) not in {
        ("single",),
        ("direct", "single"),
    }:
        parser.error("--single-executable-causal-mastery supports only direct and single families")
    if args.single_causal_mastery and args.single_executable_causal_mastery:
        parser.error("the two causal mastery modes are mutually exclusive")
    causal_mastery_modes = sum(
        bool(mode)
        for mode in (
            args.single_causal_mastery,
            args.single_executable_causal_mastery,
            args.single_resumption_control_mastery,
        )
    )
    if causal_mastery_modes > 1:
        parser.error("the single causal and resumption mastery modes are mutually exclusive")
    if args.single_resumption_control_mastery and tuple(args.families) not in {
        ("single",),
        ("direct", "single"),
    }:
        parser.error("--single-resumption-control-mastery supports only direct and single families")
    if args.single_causal_mastery and (
        args.single_control_atoms
        or args.single_exact_spawn_only
        or args.single_admission_only
        or args.extra_single_control_instances
        or args.extra_single_parent_instances
    ):
        parser.error("--single-causal-mastery cannot be mixed with other single modes")
    if args.single_executable_causal_mastery and (
        args.single_control_atoms
        or args.single_exact_spawn_only
        or args.single_admission_only
        or args.extra_single_control_instances
        or args.extra_single_parent_instances
    ):
        parser.error("--single-executable-causal-mastery cannot be mixed with other single modes")
    if args.single_resumption_control_mastery and (
        args.single_control_atoms
        or args.single_exact_spawn_only
        or args.single_admission_only
        or args.extra_single_control_instances
        or args.extra_single_parent_instances
    ):
        parser.error("--single-resumption-control-mastery cannot be mixed with other single modes")
    if args.single_exact_spawn_only and tuple(args.families) != ("single",):
        parser.error("--single-exact-spawn-only requires --families single")
    if args.single_exact_spawn_only and args.single_control_atoms:
        parser.error("--single-exact-spawn-only already selects the spawn control atom")
    if args.single_admission_only and "single" not in args.families:
        parser.error("--single-admission-only requires the single family")
    if args.single_admission_only and any(family not in {"direct", "single"} for family in args.families):
        parser.error("--single-admission-only supports only direct and single families")
    if args.single_admission_only and (
        args.single_control_atoms
        or args.single_exact_spawn_only
        or args.extra_single_control_instances
        or args.extra_single_parent_instances
    ):
        parser.error("--single-admission-only cannot be mixed with other single modes")
    if args.extra_single_control_instances < 0:
        parser.error("--extra-single-control-instances must be nonnegative")
    if args.extra_single_control_instances and not args.single_control_atoms:
        parser.error("--extra-single-control-instances requires --single-control-atoms")
    if args.extra_single_parent_instances < 0:
        parser.error("--extra-single-parent-instances must be nonnegative")
    if args.extra_single_parent_instances and "single" not in args.families:
        parser.error("--extra-single-parent-instances requires the single family")
    if args.retained_spawn_only and args.binding_focused_only:
        parser.error("--retained-spawn-only and --binding-focused-only are mutually exclusive")

    prompt = _current_system_prompt(args.harness_trace)
    taskset_config = {
        "split": "train",
        "families": tuple(args.families),
        "instruction_level": "standard",
        "instances_per_template": args.instances,
        "instance_offset": args.instance_offset,
        "seed": args.seed,
    }
    self_contained_child_contract = (
        args.self_contained_child_contract
        or args.single_executable_causal_mastery
        or args.single_resumption_control_mastery
    )
    tasks = (
        SubagentAdmissionTaskset(
            SubagentAdmissionConfig(
                **taskset_config,
                self_contained_child_contract=True,
            )
        ).load()
        if self_contained_child_contract
        else SubagentCommunicationTaskset(
            SubagentCommunicationConfig(**taskset_config)
        ).load()
    )
    examples = []
    for task in tasks:
        if args.binding_focused_only:
            examples.extend(_binding_focused_atoms(task, prompt))
            continue
        if args.retained_spawn_only:
            example = _followup_protocol_atoms(task, prompt)[0]
            example["metadata"]["family"] = "retained_spawn_atom"
            if args.thinking_rationales:
                _add_retained_spawn_thinking(example)
            examples.append(example)
            continue
        if args.single_exact_spawn_only:
            examples.append(
                next(
                    atom
                    for atom in _single_control_atoms(task, prompt)
                    if atom["metadata"]["context_style"] == "parent_spawn_exact_payload"
                )
            )
            continue
        if task.data.family == "direct":
            examples.append(_direct_example(task, prompt))
        elif task.data.family == "single":
            if args.single_resumption_control_mastery:
                examples.append(_single_admission_example(task, prompt))
                examples.append(_single_examples(task, prompt)[1])
                examples.append(
                    next(
                        atom
                        for atom in _single_control_atoms(task, prompt)
                        if atom["metadata"]["context_style"] == "parent_wait_for_reply"
                    )
                )
                recovery_atoms = _single_recovery_atoms(task, prompt)
                examples.extend(
                    atom
                    for atom in recovery_atoms
                    if atom["metadata"]["context_style"]
                    in {"child_string_type_repair", "parent_top_level_await_repair"}
                )
                examples.extend(_single_resumption_atoms(task, prompt))
            elif args.single_executable_causal_mastery:
                examples.extend(_single_examples(task, prompt))
                examples.append(_single_admission_example(task, prompt))
                examples.extend(
                    atom
                    for atom in _single_control_atoms(task, prompt)
                    if atom["metadata"]["context_style"]
                    in {
                        "child_send",
                        "parent_wait_for_reply",
                        "parent_bind_reply_and_finalize",
                    }
                )
                examples.extend(_single_recovery_atoms(task, prompt))
            elif args.single_causal_mastery:
                examples.extend(_single_examples(task, prompt))
                examples.append(_single_admission_example(task, prompt))
                examples.extend(
                    atom
                    for atom in _single_control_atoms(task, prompt)
                    if atom["metadata"]["context_style"]
                    in {
                        "child_send",
                        "parent_wait_for_reply",
                        "parent_bind_reply_and_finalize",
                    }
                )
            elif args.single_admission_only:
                examples.append(_single_admission_example(task, prompt))
            else:
                examples.extend(_single_examples(task, prompt))
            if args.single_control_atoms and not args.single_admission_only and not args.single_causal_mastery:
                examples.extend(_single_control_atoms(task, prompt))
        elif task.data.family == "parallel":
            examples.extend(_parallel_examples(task, prompt))
            if args.parallel_control_atoms:
                examples.extend(_parallel_control_atoms(task, prompt))
        elif task.data.family == "followup":
            for copy in range(args.followup_copies):
                copied = _followup_examples(task, prompt)
                for example in copied:
                    example["metadata"]["copy"] = copy
                examples.extend(copied)
            if args.protocol_atoms:
                examples.extend(_followup_protocol_atoms(task, prompt))
            if args.binding_focused_atoms:
                examples.extend(_binding_focused_atoms(task, prompt))
            if args.bidirectional_request_atoms:
                for copy in range(args.bidirectional_request_copies):
                    atom = _bidirectional_request_atom(task, prompt)
                    atom["metadata"]["copy"] = copy
                    examples.append(atom)
        else:
            for copy in range(args.handshake_copies):
                copied = _handshake_examples(task, prompt)
                for example in copied:
                    example["metadata"]["copy"] = copy
                examples.extend(copied)
            if args.bidirectional_request_atoms:
                for copy in range(args.bidirectional_request_copies):
                    atom = _bidirectional_request_atom(task, prompt)
                    atom["metadata"]["copy"] = copy
                    examples.append(atom)

    if args.extra_binding_instances:
        extra_tasks = SubagentCommunicationTaskset(
            SubagentCommunicationConfig(
                split="train",
                families=("followup",),
                instruction_level="standard",
                instances_per_template=args.extra_binding_instances,
                instance_offset=args.instance_offset + 10_000,
                seed=args.seed + 1,
            )
        ).load()
        for task in extra_tasks:
            atoms = _binding_focused_atoms(task, prompt)
            for atom in atoms:
                atom["metadata"]["source"] = "extra_binding"
            examples.extend(atoms)

    if args.extra_parallel_control_instances:
        extra_tasks = SubagentCommunicationTaskset(
            SubagentCommunicationConfig(
                split="train",
                families=("parallel",),
                instruction_level="standard",
                instances_per_template=args.extra_parallel_control_instances,
                instance_offset=args.instance_offset + 20_000,
                seed=args.seed + 2,
            )
        ).load()
        for task in extra_tasks:
            atoms = _parallel_control_atoms(task, prompt)
            for atom in atoms:
                atom["metadata"]["source"] = "extra_parallel_control"
            examples.extend(atoms)

    if args.extra_parallel_parent_instances:
        extra_tasks = SubagentCommunicationTaskset(
            SubagentCommunicationConfig(
                split="train",
                families=("parallel",),
                instruction_level="standard",
                instances_per_template=args.extra_parallel_parent_instances,
                instance_offset=args.instance_offset + 30_000,
                seed=args.seed + 3,
            )
        ).load()
        for task in extra_tasks:
            parent = _parallel_examples(task, prompt)[0]
            parent["metadata"]["source"] = "extra_parallel_parent"
            examples.append(parent)

    if args.extra_single_control_instances:
        extra_tasks = SubagentCommunicationTaskset(
            SubagentCommunicationConfig(
                split="train",
                families=("single",),
                instruction_level="standard",
                instances_per_template=args.extra_single_control_instances,
                instance_offset=args.instance_offset + 40_000,
                seed=args.seed + 4,
            )
        ).load()
        for task in extra_tasks:
            atoms = _single_control_atoms(task, prompt)
            for atom in atoms:
                atom["metadata"]["source"] = "extra_single_control"
            examples.extend(atoms)

    if args.extra_single_parent_instances:
        extra_tasks = SubagentCommunicationTaskset(
            SubagentCommunicationConfig(
                split="train",
                families=("single",),
                instruction_level="standard",
                instances_per_template=args.extra_single_parent_instances,
                instance_offset=args.instance_offset + 50_000,
                seed=args.seed + 5,
            )
        ).load()
        for task in extra_tasks:
            parent = _single_examples(task, prompt)[0]
            parent["metadata"]["source"] = "extra_single_parent"
            examples.append(parent)

    if (
        args.extra_binding_instances
        or args.extra_parallel_control_instances
        or args.extra_parallel_parent_instances
        or args.extra_single_control_instances
        or args.extra_single_parent_instances
    ):
        for example in examples:
            example["metadata"].setdefault("source", "primary")

    _validate_no_repeated_tool_calls(examples)
    _validate_event_driven_parent_turns(examples)
    _validate_followup_message_binding(examples)
    _validate_handshake_message_binding(examples)
    _validate_parent_child_handle_continuity(examples)
    _validate_retained_spawn_atoms(examples)
    _validate_retained_spawn_thinking(examples)
    _validate_binding_spawn_atoms(examples)
    _validate_single_control_atoms(examples)
    _validate_single_recovery_atoms(examples)
    _validate_single_resumption_atoms(examples)
    _validate_single_admission_trajectories(examples)
    _validate_parallel_control_atoms(examples)
    _validate_parallel_parent_bindings(examples)
    _validate_atomic_local_computation(examples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(example) + "\n" for example in examples))
    print(f"wrote {len(examples)} examples for {','.join(args.families)} to {args.output}")


if __name__ == "__main__":
    main()
