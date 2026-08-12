import json
import re
import subprocess
import sys
from pathlib import Path


def test_followup_child_binds_multiplier_from_parent_message(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "followup",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    children = [example for example in examples if example["metadata"]["role"] == "child"]
    assert children
    for child in children:
        messages = child["messages"]
        parent_index = next(
            index
            for index, message in enumerate(messages)
            if message["role"] == "user" and message["content"].startswith("[from parent]")
        )
        body = messages[parent_index]["content"].rsplit("\n\n", 1)[-1]
        result_call = next(call for message in messages[parent_index + 1 :] for call in message.get("tool_calls", []))
        code = json.loads(result_call["function"]["arguments"])["code"]

        assert f"parent_message_body = {body!r}" in code
        assert "multiplier = int(parent_message_body.strip())" in code
        assert not re.search(r"(?m)^multiplier\s*=\s*-?\d+\s*$", code)


def test_handshake_export_preserves_both_message_directions(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "handshake",
            "--handshake-copies",
            "2",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(examples) == 16
    assert {example["metadata"]["role"] for example in examples} == {"parent", "child"}
    assert {example["metadata"]["copy"] for example in examples} == {0, 1}

    for child in [example for example in examples if example["metadata"]["role"] == "child"]:
        messages = child["messages"]
        parent_index = next(
            index
            for index, message in enumerate(messages)
            if message["role"] == "user" and message["content"].startswith("[from parent]")
        )
        body = messages[parent_index]["content"].rsplit("\n\n", 1)[-1]
        result_call = next(
            call for message in messages[parent_index + 1 :] for call in message.get("tool_calls", [])
        )
        code = json.loads(result_call["function"]["arguments"])["code"]

        assert f"parent_message_body = {body!r}" in code
        assert "nonce = int(parent_message_body.strip())" in code
        assert not re.search(r"(?m)^nonce\s*=\s*\d+\s*$", code)


def test_bidirectional_request_atoms_end_after_exact_child_request(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "followup",
            "handshake",
            "--bidirectional-request-atoms",
            "--bidirectional-request-copies",
            "2",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    atoms = [example for example in examples if example["metadata"]["family"] == "bidirectional_request_atom"]
    assert len(atoms) == 16
    assert {atom["metadata"]["source_family"] for atom in atoms} == {"followup", "handshake"}

    for atom in atoms:
        codes = [
            json.loads(call["function"]["arguments"])["code"]
            for message in atom["messages"]
            for call in message.get("tool_calls", [])
        ]
        expected = "'need multiplier'" if atom["metadata"]["source_family"] == "followup" else "'need nonce'"
        sends = [code for code in codes if "agent_message.send" in code]
        assert len(sends) == 1
        assert expected in sends[0]
        assert atom["messages"][-1]["role"] == "assistant"
        assert "Requested the" in atom["messages"][-1]["content"]
        assert not any(
            message["role"] == "user" and message["content"].startswith("[from parent]")
            for message in atom["messages"]
        )


def test_delegated_parents_yield_for_messages_without_polling(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "single",
            "parallel",
            "followup",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    parents = [example for example in examples if example["metadata"]["role"] == "parent"]
    assert parents
    for parent in parents:
        messages = parent["messages"]
        incoming = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "user" and message["content"].startswith("[from child:")
        ]
        assert incoming
        assert all(messages[index - 1]["role"] == "assistant" for index in incoming)
        codes = [
            json.loads(call["function"]["arguments"])["code"]
            for message in messages
            for call in message.get("tool_calls", [])
        ]
        assert not any("agent_observe" in code for code in codes)


def test_harness_trace_uses_current_environment_prompt(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "nodes": [
                            {
                                "message": {
                                    "role": "system",
                                    "content": (
                                        "Prime Agent base prompt.\n\n"
                                        "Coordinate work through Prime Agent's persistent IPython "
                                        "kernel. Obsolete bounded agent_observe polling policy."
                                    ),
                                }
                            }
                        ]
                    }
                ]
            }
        )
        + "\n"
    )
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "followup",
            "--harness-trace",
            str(trace),
        ],
        check=True,
    )

    example = json.loads(output.read_text().splitlines()[0])
    prompt = example["messages"][0]["content"]
    assert prompt.startswith("Prime Agent base prompt.")
    assert "end the current turn without polling" in prompt
    assert "Obsolete bounded agent_observe polling policy" not in prompt


def test_protocol_atoms_teach_only_corrected_native_operations(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "followup",
            "--protocol-atoms",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    atoms = [example for example in examples if example["metadata"]["family"] == "protocol_atom"]
    assert len(atoms) == 20
    trained_code = [
        json.loads(call["function"]["arguments"])["code"]
        for example in atoms
        for message in example["messages"]
        for call in message.get("tool_calls", [])
    ]
    assert any("multiplier = " in code and "child = await rlm(" in code for code in trained_code)
    assert any("receiver_role='child'" in code for code in trained_code)
    assert any("receiver_role='parent'" in code for code in trained_code)
    assert not any("await child =" in code or "await _ =" in code for code in trained_code)
    assert not any("agent_observe" in code or "agent_message(" in code for code in trained_code)


def test_followup_parent_retains_multiplier_before_yielding(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "followup",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    parents = [example for example in examples if example["metadata"]["role"] == "parent"]
    assert parents
    for parent in parents:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in parent["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert re.search(r"(?m)^multiplier = \d+$", calls[0])
        assert "child = await rlm(" in calls[0]
        assert "agent_message.send(str(multiplier)" in calls[1]
        assert not re.search(r"(?m)^multiplier = \d+$", calls[1])


def test_protocol_followup_retains_child_handle_before_direct_send(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "followup",
            "--protocol-atoms",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    parent_atoms = [
        example
        for example in examples
        if example["metadata"]["family"] == "protocol_atom" and example["metadata"]["role"] == "parent"
    ]
    send_atom = next(
        example
        for example in parent_atoms
        if any(
            "receiver_name=child.name" in json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        )
        and not any("active as variable child" in str(message.get("content", "")) for message in example["messages"])
    )
    codes = [
        json.loads(call["function"]["arguments"])["code"]
        for message in send_atom["messages"]
        for call in message.get("tool_calls", [])
    ]
    assert "multiplier = " in codes[0]
    assert "child = await rlm(" in codes[0]
    assert "receiver_name=child.name" in codes[1]


def test_retained_spawn_only_isolates_varied_handle_binding_decisions(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "2",
            "--instance-offset",
            "900",
            "--families",
            "followup",
            "--retained-spawn-only",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(examples) == 8
    assert {example["metadata"]["family"] for example in examples} == {"retained_spawn_atom"}
    assert len({example["metadata"]["task"] for example in examples}) == 8
    codes = []
    for example in examples:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert len(calls) == 1
        assert re.search(r"(?m)^multiplier = -?\d+$", calls[0])
        assert "child = await rlm(" in calls[0]
        assert "agent_message" not in calls[0]
        codes.extend(calls)
    assert len(set(codes)) == len(codes)


def test_retained_spawn_thinking_rationales_encode_ownership_and_yield(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "followup",
            "--retained-spawn-only",
            "--thinking-rationales",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(examples) == 4
    for example in examples:
        assert example["metadata"]["thinking_rationale"] is True
        assistant_messages = [
            message for message in example["messages"] if message["role"] == "assistant"
        ]
        assert len(assistant_messages) == 2
        reasoning = " ".join(message["reasoning_content"] for message in assistant_messages)
        assert "coordinator-owned" in reasoning
        assert "child-owned" in reasoning
        assert "child handle" in reasoning
        assert "instead of polling" in reasoning


def test_thinking_rationales_require_retained_spawn_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(tmp_path / "train.json"),
            "--families",
            "followup",
            "--thinking-rationales",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires --retained-spawn-only" in result.stderr


def test_binding_focused_only_targets_short_handle_assignment_after_retained_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--instance-offset",
            "940",
            "--families",
            "followup",
            "--binding-focused-only",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(examples) == 16
    assert {example["metadata"]["family"] for example in examples} == {"binding_spawn_atom"}
    assert {example["metadata"]["context_style"] for example in examples} == {
        "retained_state",
        "correct_bare_spawn",
        "correct_missing_await",
        "silent_cell_boundary",
    }
    for example in examples:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert calls[-1] == "child = await rlm(spawn_prompt, name=child_name)"
        assert not any(re.search(r"(?m)^await rlm\(", code) for code in calls)
        assert not any("agent_message" in code or "agent_observe" in code for code in calls)


def test_binding_focused_only_rejects_incompatible_modes(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(tmp_path / "train.json"),
            "--families",
            "followup",
            "--retained-spawn-only",
            "--binding-focused-only",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_binding_focused_atoms_mix_with_complete_followup_traces(tmp_path: Path) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "followup",
            "--binding-focused-atoms",
            "--extra-binding-instances",
            "1",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    families = [example["metadata"]["family"] for example in examples]
    assert families.count("followup") == 8
    assert families.count("binding_spawn_atom") == 32
    assert sum(example["metadata"].get("source") == "extra_binding" for example in examples) == 16


def test_parallel_control_atoms_isolate_child_delivery_and_parent_fan_in(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "parallel",
            "--parallel-control-atoms",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    atoms = [example for example in examples if example["metadata"]["family"] == "parallel_control_atom"]
    assert len(atoms) == 24
    assert {example["metadata"]["context_style"] for example in atoms} == {
        "child_send",
        "parent_local_compute",
        "parent_wait_for_both",
        "parent_bind_first_and_wait",
        "parent_bind_second_and_fan_in",
    }
    child_atoms = [example for example in atoms if example["metadata"]["role"] == "child"]
    assert len(child_atoms) == 8
    for atom in child_atoms:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in atom["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert calls == ["await agent_message.send(str(checksum), receiver_role='parent')"]

    parent_atoms = [example for example in atoms if example["metadata"]["role"] == "parent"]
    assert len(parent_atoms) == 16
    calls_by_style = {
        style: [
            json.loads(call["function"]["arguments"])["code"]
            for example in parent_atoms
            if example["metadata"]["context_style"] == style
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        for style in {
            "parent_local_compute",
            "parent_wait_for_both",
            "parent_bind_first_and_wait",
            "parent_bind_second_and_fan_in",
        }
    }
    assert calls_by_style["parent_wait_for_both"] == []
    assert all("local = sum((index + 1) * value" in code for code in calls_by_style["parent_local_compute"])
    assert all("_message_body" in code and " = int(" in code for code in calls_by_style["parent_bind_first_and_wait"])
    assert all(
        "_message_body" in code and "'total': local + alpha + beta" in code
        for code in calls_by_style["parent_bind_second_and_fan_in"]
    )
    assert all(
        not any(forbidden in code for forbidden in ("agent_observe", "asyncio.sleep", "ai.get_harness_state"))
        for calls in calls_by_style.values()
        for code in calls
    )


def test_extra_parallel_control_instances_are_unique_and_schema_stable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "parallel",
            "followup",
            "--parallel-control-atoms",
            "--binding-focused-atoms",
            "--extra-binding-instances",
            "1",
            "--extra-parallel-control-instances",
            "1",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    extra = [example for example in examples if example["metadata"].get("source") == "extra_parallel_control"]
    assert len(extra) == 24
    assert len({(example["metadata"]["task"], json.dumps(example["messages"])) for example in extra}) == 24
    assert all("source" in example["metadata"] for example in examples)


def test_extra_parallel_parent_instances_replay_complete_semantic_fan_in(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "parallel",
            "--extra-parallel-parent-instances",
            "1",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    extra = [example for example in examples if example["metadata"].get("source") == "extra_parallel_parent"]
    assert len(extra) == 4
    assert len({example["metadata"]["task"] for example in extra}) == 4
    for parent in extra:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in parent["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert sum(" = await rlm(" in code for code in calls) == 2
        local_calls = [code for code in calls if "local_values =" in code]
        assert len(local_calls) == 1
        assert "local = sum((index + 1) * value" in local_calls[0]
        incoming = [
            (index, message)
            for index, message in enumerate(parent["messages"])
            if message["role"] == "user" and str(message.get("content", "")).startswith("[from child:")
        ]
        assert len(incoming) == 2
        for index, message in incoming:
            key = message["content"].split("[from child:", 1)[1].split("-worker]", 1)[0]
            body = message["content"].rsplit("\n\n", 1)[-1]
            call = parent["messages"][index + 1]["tool_calls"][0]
            code = json.loads(call["function"]["arguments"])["code"]
            assert f"{key}_message_body = {body!r}" in code
            assert f"{key} = int({key}_message_body.strip())" in code
        final_calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in parent["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert "'total': local + alpha + beta" in final_calls[-1]
        answer = json.loads(parent["messages"][-1]["content"])
        assert set(answer) == {"local", "alpha", "beta", "total"}
        assert answer["total"] == answer["local"] + answer["alpha"] + answer["beta"]


def test_single_control_atoms_isolate_exact_payload_and_event_driven_fan_in(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "single",
            "--single-control-atoms",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    atoms = [example for example in examples if example["metadata"]["family"] == "single_control_atom"]
    assert len(atoms) == 20
    assert {example["metadata"]["context_style"] for example in atoms} == {
        "child_send",
        "parent_spawn_exact_payload",
        "parent_local_compute",
        "parent_wait_for_reply",
        "parent_bind_reply_and_finalize",
    }
    for atom in atoms:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in atom["messages"]
            for call in message.get("tool_calls", [])
        ]
        style = atom["metadata"]["context_style"]
        if style == "child_send":
            assert atom["metadata"]["role"] == "child"
            assert calls == ["await agent_message.send(str(checksum), receiver_role='parent')"]
            assert atom["messages"][-1] == {
                "role": "assistant",
                "content": "Sent the checksum to the parent.",
            }
        elif style == "parent_spawn_exact_payload":
            formula = "sum((index + 1) * value for index, value in enumerate(values))"
            assert calls == [
                "handle = await rlm("
                + repr(
                    f"Read {atom['metadata']['path']} as the JSON list values and compute its "
                    f"weighted checksum using exactly {formula}. Then execute exactly once: "
                    "await agent_message.send(str(checksum), receiver_role='parent'). After the "
                    "delivery receipt succeeds, answer with a concise confirmation and stop."
                )
                + ", name='shard-worker')"
            ]
        assert not any(forbidden in code for code in calls for forbidden in ("agent_observe", "asyncio", "sleep("))


def test_single_causal_mastery_balances_full_traces_and_ordered_transitions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "single-causal-mastery.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--families",
            "direct",
            "single",
            "--instances",
            "1",
            "--single-causal-mastery",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    styles: dict[str, int] = {}
    for example in examples:
        if example["metadata"]["family"] == "direct":
            continue
        style = example["metadata"].get("context_style", "full")
        styles[style] = styles.get(style, 0) + 1

    assert len(examples) == 28
    assert styles == {
        "full": 8,
        "exact_spawn_local_yield": 4,
        "child_send": 4,
        "parent_wait_for_reply": 4,
        "parent_bind_reply_and_finalize": 4,
    }


def test_extra_single_examples_concentrate_parent_control_without_duplicate_tasks(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "direct",
            "single",
            "parallel",
            "--single-control-atoms",
            "--extra-single-control-instances",
            "1",
            "--extra-single-parent-instances",
            "1",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    extra_control = [example for example in examples if example["metadata"].get("source") == "extra_single_control"]
    extra_parents = [example for example in examples if example["metadata"].get("source") == "extra_single_parent"]
    assert len(extra_control) == 20
    assert len(extra_parents) == 4
    assert len({example["metadata"]["task"] for example in extra_parents}) == 4
    assert {example["metadata"]["task"] for example in extra_control}.isdisjoint(
        {example["metadata"]["task"] for example in extra_parents}
    )
    assert all(example["metadata"].get("source") for example in examples)


def test_single_exact_spawn_only_varies_paths_without_diluting_the_action(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "2",
            "--families",
            "single",
            "--single-exact-spawn-only",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(examples) == 8
    assert {example["metadata"]["context_style"] for example in examples} == {"parent_spawn_exact_payload"}
    assert len({example["metadata"]["path"] for example in examples}) == 8
    for example in examples:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert len(calls) == 1
        assert example["metadata"]["path"] in calls[0]
        assert calls[0].startswith("handle = await rlm(")


def test_single_admission_only_keeps_natural_prefix_and_direct_replay(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "2",
            "--families",
            "direct",
            "single",
            "--single-admission-only",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    direct = [example for example in examples if example["metadata"]["family"] == "direct"]
    admission = [example for example in examples if example["metadata"]["family"] == "single_admission_trajectory"]
    assert len(direct) == 8
    assert len(admission) == 8
    for example in admission:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        path = example["metadata"]["path"]
        assert len(calls) == 2
        assert calls[0].startswith("handle = await rlm(")
        assert path in calls[0]
        assert "agent_message" in calls[0]
        assert path not in calls[1]
        assert "local = sum(" in calls[1]
        assert not any(
            message["role"] == "user" and message["content"].startswith("[from child:shard-worker]")
            for message in example["messages"]
        )
        assert example["messages"][-1]["content"] == "Waiting for shard-worker's explicit reply."


def test_single_admission_can_match_the_self_contained_environment_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--instances",
            "1",
            "--families",
            "direct",
            "single",
            "--single-admission-only",
            "--self-contained-child-contract",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    direct = [example for example in examples if example["metadata"]["family"] == "direct"]
    admission = [
        example
        for example in examples
        if example["metadata"]["family"] == "single_admission_trajectory"
    ]
    assert direct
    assert admission
    assert all("Delegation contract:" not in example["messages"][1]["content"] for example in direct)
    for example in admission:
        prompt = example["messages"][1]["content"]
        assert "Delegation contract:" in prompt
        assert "sum((index + 1) * value for index, value in enumerate(values))" in prompt
        spawn = json.loads(example["messages"][2]["tool_calls"][0]["function"]["arguments"])["code"]
        assert "sum((index + 1) * value for index, value in enumerate(values))" in spawn


def test_single_executable_causal_mastery_covers_live_recovery_boundaries(
    tmp_path: Path,
) -> None:
    output = tmp_path / "single-executable-causal-mastery.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--families",
            "direct",
            "single",
            "--instances",
            "1",
            "--single-executable-causal-mastery",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    direct = [example for example in examples if example["metadata"]["family"] == "direct"]
    recovery = [
        example for example in examples if example["metadata"]["family"] == "single_recovery_atom"
    ]
    admission = [
        example
        for example in examples
        if example["metadata"]["family"] == "single_admission_trajectory"
    ]

    assert len(examples) == 40
    assert len(direct) == 4
    assert len(admission) == 4
    assert len(recovery) == 12
    assert {example["metadata"]["context_style"] for example in recovery} == {
        "child_string_type_repair",
        "child_declared_tool_repair",
        "parent_top_level_await_repair",
    }
    assert all("Delegation contract:" not in example["messages"][1]["content"] for example in direct)
    assert all("Delegation contract:" in example["messages"][1]["content"] for example in admission)

    for example in recovery:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        assert len(calls) == 1
        assert "run_until_complete" not in calls[0]
        if example["metadata"]["role"] == "child":
            assert calls == ["await agent_message.send(str(checksum), receiver_role='parent')"]
            assert example["messages"][-1]["content"] == "Sent the checksum to the parent."
        else:
            assert calls[0].startswith("handle = await rlm(")
            assert example["metadata"]["path"] in calls[0]


def test_single_resumption_control_mastery_targets_noop_boundaries(
    tmp_path: Path,
) -> None:
    output = tmp_path / "single-resumption-control-mastery.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_subagent_communication_sft.py",
            str(output),
            "--families",
            "direct",
            "single",
            "--instances",
            "1",
            "--single-resumption-control-mastery",
        ],
        check=True,
    )

    examples = [json.loads(line) for line in output.read_text().splitlines()]
    direct = [example for example in examples if example["metadata"]["family"] == "direct"]
    resumption = [
        example for example in examples if example["metadata"]["family"] == "single_resumption_atom"
    ]

    assert len(examples) == 40
    assert len(direct) == 4
    assert len(resumption) == 16
    assert {example["metadata"]["context_style"] for example in resumption} == {
        "parent_inert_wait_repair",
        "parent_message_finalize_without_tool",
        "child_inert_send_repair",
        "child_stop_after_receipt",
    }

    for example in resumption:
        calls = [
            json.loads(call["function"]["arguments"])["code"]
            for message in example["messages"]
            for call in message.get("tool_calls", [])
        ]
        style = example["metadata"]["context_style"]
        if style == "child_inert_send_repair":
            assert calls == ["await agent_message.send(str(checksum), receiver_role='parent')"]
        else:
            assert calls == []
        if style == "parent_inert_wait_repair":
            assert example["messages"][-1]["content"] == "Waiting for shard-worker's explicit reply."
        elif style == "parent_message_finalize_without_tool":
            assert example["messages"][-1]["content"] == json.dumps(example["metadata"]["answer"])
        elif style == "child_stop_after_receipt":
            assert example["messages"][-1]["content"] == "Sent the checksum to the parent."
