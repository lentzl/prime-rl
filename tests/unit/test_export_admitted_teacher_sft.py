import json
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage

from scripts.export_admitted_teacher_sft import (
    CONTRACT_MARKER,
    _admitted,
    _branch_role,
    _executable_branches,
    _normalize_text_content,
    _repairable_final,
    _strip_teacher_demonstration,
    export,
)


def _raw_trace(trace_id: str, admitted: bool) -> dict:
    call = ToolCall(id="call-1", name="ipython", arguments=json.dumps({"code": "value = 1"}))
    nodes = [
        MessageNode(parent=None, message=SystemMessage(content="system"), sampled=False),
        MessageNode(
            parent=0,
            message=UserMessage(
                content=(
                    "<Question>\nsolve this\n"
                    "This is an example for a response to the question:\n"
                    "<Demonstration>\ngolden answer\n"
                    "Now answer with a response of your own."
                    f"{CONTRACT_MARKER} include the formula."
                )
            ),
            sampled=False,
        ),
        MessageNode(parent=1, message=AssistantMessage(content=None, tool_calls=[call]), sampled=True),
        MessageNode(parent=2, message=ToolMessage(tool_call_id="call-1", content=""), sampled=False),
        MessageNode(parent=3, message=AssistantMessage(content='{"answer": 1}'), sampled=True),
    ]
    trace = vf.Trace(
        id=trace_id,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, name="task")),
        nodes=nodes,
    ).model_dump(mode="json", exclude_none=True)
    trace["task"]["data"]["answer"] = {"answer": 1}
    score = 1.0 if admitted else 0.8
    trace.update(
        {
            "ok": True,
            "metrics": {
                "causal_completed": 1.0 if admitted else 0.0,
                "causal_clean_completed": 1.0 if admitted else 0.0,
                "answer_accuracy": 1.0,
                "protocol_aligned": 1.0,
                "duplicate_cells": 0.0,
                "post_fan_in_failed_cells": 0.0,
                "post_fan_in_control_aligned": 1.0,
            },
            "rewards": {"causal_control": {"score": score, "weight": 1.0}},
        }
    )
    return trace


def test_admission_requires_final_metrics_and_clean_control() -> None:
    assert _admitted(_raw_trace("accepted", True))
    assert not _admitted(_raw_trace("partial", False))


def test_admission_rejects_failed_post_fan_in_control() -> None:
    trace = _raw_trace("noisy", True)
    trace["metrics"]["post_fan_in_failed_cells"] = 1.0
    trace["metrics"]["post_fan_in_control_aligned"] = 0.0

    assert not _admitted(trace)


def test_admission_rejects_undeclared_failed_tool_call() -> None:
    trace = _raw_trace("fabricated-tool", True)
    trace["nodes"][2]["message"]["tool_calls"][0]["name"] = "agent_message"
    trace["nodes"][3]["message"]["content"] = "Tool agent_message not found"

    assert not _admitted(trace)


def test_admission_rejects_failed_declared_tool_result() -> None:
    trace = _raw_trace("failed-ipython", True)
    trace["nodes"][3]["message"]["content"] = "Traceback (most recent call last)\nNameError: value"

    assert not _admitted(trace)


def _child_trace(*, post_send_tool: bool) -> dict:
    trace = _raw_trace("child", True)
    trace["nodes"][1]["message"]["content"] = "[task from parent]\ncompute and reply"
    trace["nodes"][2]["message"]["tool_calls"][0]["arguments"] = json.dumps(
        {"code": "await agent_message.send('1', receiver_role='parent')"}
    )
    trace["nodes"][3]["message"]["content"] = "{'deliveryStatus': 'queued'}"
    if not post_send_tool:
        return trace

    call = ToolCall(id="call-2", name="ipython", arguments=json.dumps({"code": "print('waiting')"}))
    extra_nodes = [
        MessageNode(parent=3, message=AssistantMessage(content=None, tool_calls=[call]), sampled=True),
        MessageNode(parent=4, message=ToolMessage(tool_call_id="call-2", content="waiting"), sampled=False),
    ]
    final = trace["nodes"][-1]
    final["parent"] = 5
    trace["nodes"] = (
        trace["nodes"][:4] + [node.model_dump(mode="json", exclude_none=True) for node in extra_nodes] + [final]
    )
    return trace


def test_executable_branches_accepts_child_that_stops_after_send() -> None:
    assert _executable_branches(_child_trace(post_send_tool=False))


def test_executable_branches_rejects_child_tool_use_after_send() -> None:
    assert not _executable_branches(_child_trace(post_send_tool=True))


def test_executable_branches_rejects_coordinator_polling_inside_ipython() -> None:
    for code in (
        "import time\ntime.sleep(2)",
        "await agent_message.list_agents()",
        "await rlm.list_subagents()",
        "await agent_observe.get_agent(name='worker')",
        "session_dir = '/tmp/agent/sessions/run'\nos.listdir(session_dir)",
        "print('Waiting for child reply...')",
    ):
        trace = _raw_trace("polling", True)
        trace["nodes"][2]["message"]["tool_calls"][0]["arguments"] = json.dumps({"code": code})

        assert not _executable_branches(trace), code


def test_executable_branches_rejects_noop_ipython_cells() -> None:
    for code in ("# No tool call - end the turn", "pass", "'Waiting for child reply.'"):
        trace = _raw_trace("noop", True)
        trace["nodes"][2]["message"]["tool_calls"][0]["arguments"] = json.dumps({"code": code})

        assert not _executable_branches(trace), code


def test_executable_branches_rejects_tools_between_waiting_turn_and_child_reply() -> None:
    trace = _raw_trace("wait-then-tool", True)
    trace["nodes"][2]["message"]["tool_calls"][0]["arguments"] = json.dumps(
        {"code": "handle = await rlm('work', name='worker')"}
    )
    waiting = MessageNode(
        parent=3,
        message=AssistantMessage(content="Waiting for the child's explicit reply."),
        sampled=True,
    )
    call = ToolCall(id="call-2", name="ipython", arguments=json.dumps({"code": "value += 1"}))
    polling = MessageNode(parent=4, message=AssistantMessage(content=None, tool_calls=[call]), sampled=True)
    result = MessageNode(parent=5, message=ToolMessage(tool_call_id="call-2", content=""), sampled=False)
    final = trace["nodes"][-1]
    final["parent"] = 6
    trace["nodes"] = (
        trace["nodes"][:4]
        + [node.model_dump(mode="json", exclude_none=True) for node in (waiting, polling, result)]
        + [final]
    )

    assert not _executable_branches(trace)


def test_executable_branches_rejects_gate_retry_after_child_reply() -> None:
    trace = _raw_trace("late-gate-retry", True)
    trace["nodes"][2]["message"]["tool_calls"][0]["arguments"] = json.dumps(
        {"code": "handle = await rlm('work', name='worker')"}
    )
    child_reply = MessageNode(
        parent=3,
        message=UserMessage(content="[from child:worker]\n\n1"),
        sampled=False,
    )
    draft = MessageNode(parent=4, message=AssistantMessage(content="The answer is 1."), sampled=True)
    retry = MessageNode(
        parent=5,
        message=UserMessage(content="Autonomous quality gate failed (attempt 2/12): strict JSON required"),
        sampled=False,
    )
    final = trace["nodes"][-1]
    final["parent"] = 6
    trace["nodes"] = (
        trace["nodes"][:4]
        + [node.model_dump(mode="json", exclude_none=True) for node in (child_reply, draft, retry)]
        + [final]
    )

    assert not _executable_branches(trace)


def test_executable_branches_accepts_plain_waiting_turn() -> None:
    trace = _raw_trace("plain-wait", True)
    trace["nodes"][2]["message"]["tool_calls"][0]["arguments"] = json.dumps(
        {"code": "handle = await rlm('work', name='worker')"}
    )
    trace["nodes"][-1]["message"]["content"] = "Waiting for the child's explicit reply."

    assert _executable_branches(trace)


def test_strip_teacher_demonstration_preserves_contract() -> None:
    raw = _raw_trace("accepted", True)
    content = raw["nodes"][1]["message"]["content"]

    stripped = _strip_teacher_demonstration(content)

    assert stripped.startswith("solve this")
    assert "golden answer" not in stripped
    assert "include the formula" in stripped


def test_strip_teacher_demonstration_handles_typed_text_parts() -> None:
    raw = _raw_trace("accepted", True)
    content = [{"type": "text", "text": raw["nodes"][1]["message"]["content"]}]

    stripped = _strip_teacher_demonstration(content)

    assert stripped[0]["text"].startswith("solve this")
    assert "golden answer" not in stripped[0]["text"]
    assert "include the formula" in stripped[0]["text"]


def test_branch_role_handles_typed_text_parts() -> None:
    child_messages = [{"role": "user", "content": [{"type": "text", "text": "[task from parent]\nwork"}]}]

    assert _branch_role(child_messages) == "child"


def test_normalize_text_content_flattens_typed_parts() -> None:
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": " second"},
    ]

    assert _normalize_text_content(content) == "first second"


def test_export_writes_only_admitted_traces(tmp_path: Path) -> None:
    source = tmp_path / "traces.jsonl"
    source.write_text(
        json.dumps({"traces": [_raw_trace("accepted", True)]})
        + "\n"
        + json.dumps({"traces": [_raw_trace("partial", False)]})
        + "\n"
    )

    manifest = export([source], tmp_path / "dataset", min_traces=1)

    examples = [json.loads(line) for line in (tmp_path / "dataset" / "train.json").read_text().splitlines()]
    assert manifest["selection"]["strict"]["causal_clean_completed"] == 1.0
    assert manifest["selection"]["strict"]["sampled_tool_calls_declared"]
    assert manifest["selection"]["strict"]["tool_results_executable"]
    assert manifest["selection"]["strict"]["child_stops_tools_after_message_send"]
    assert manifest["selection"]["strict"]["coordinator_has_no_polling_cells"]
    assert manifest["accepted_trace_ids"] == ["accepted"]
    assert manifest["num_branch_examples"] == 1
    assert len(examples) == 1
    assert examples[0]["metadata"]["role"] == "coordinator"
    assert "golden answer" not in examples[0]["messages"][1]["content"]
    assert all(
        message.get("content") is None or isinstance(message["content"], str) for message in examples[0]["messages"]
    )


def test_export_can_remove_only_inert_parent_cells_for_teacher_bootstrap(tmp_path: Path) -> None:
    source = tmp_path / "traces.jsonl"
    trace = _raw_trace("inert-bootstrap", True)
    trace["nodes"][2]["message"]["tool_calls"][0]["arguments"] = json.dumps(
        {"code": "# End the turn without a tool call"}
    )
    source.write_text(json.dumps({"traces": [trace]}) + "\n")

    manifest = export(
        [source],
        tmp_path / "dataset",
        min_traces=1,
        drop_inert_coordinator_cells=True,
    )

    example = json.loads((tmp_path / "dataset" / "train.json").read_text())
    assert manifest["accepted_trace_ids"] == ["inert-bootstrap"]
    assert manifest["inert_sanitized_trace_ids"] == ["inert-bootstrap"]
    assert example["metadata"]["inert_coordinator_cells_removed"] == 1
    assert not any(message.get("tool_calls") for message in example["messages"])
    assert not any(message.get("role") == "tool" for message in example["messages"])


def test_inert_bootstrap_does_not_remove_real_polling(tmp_path: Path) -> None:
    source = tmp_path / "traces.jsonl"
    trace = _raw_trace("polling", True)
    trace["nodes"][2]["message"]["tool_calls"][0]["arguments"] = json.dumps(
        {"code": "await agent_message.list_agents()"}
    )
    source.write_text(json.dumps({"traces": [trace]}) + "\n")

    manifest = export(
        [source],
        tmp_path / "dataset",
        min_traces=0,
        drop_inert_coordinator_cells=True,
    )

    assert manifest["num_traces"] == 0
    assert manifest["inert_sanitized_trace_ids"] == []


def _repairable_trace(trace_id: str, answer: int = 1) -> dict:
    trace = _raw_trace(trace_id, admitted=True)
    trace["metrics"].update(
        {
            "causal_completed": 0.0,
            "causal_clean_completed": 0.0,
            "causal_child_reply": 1.0,
            "causal_clean_child_reply": 1.0,
            "clean_protocol_aligned": 1.0,
            "failed_cells": 0.0,
        }
    )
    trace["metrics"]["answer_accuracy"] = 0.0
    trace["task"]["data"]["answer"] = {"answer": answer}
    trace["nodes"][-1]["message"]["content"] = f'The result is ```json\n{{"answer": {answer}}}\n```'
    return trace


def test_repairable_final_requires_exact_embedded_answer() -> None:
    assert _repairable_final(_repairable_trace("repairable"))
    assert not _repairable_final(
        _repairable_trace("wrong", answer=2) | {"nodes": _repairable_trace("wrong", answer=1)["nodes"]}
    )


def test_export_canonicalizes_only_final_coordinator_text(tmp_path: Path) -> None:
    source = tmp_path / "traces.jsonl"
    source.write_text(json.dumps({"traces": [_repairable_trace("repairable")]}) + "\n")

    manifest = export(
        [source],
        tmp_path / "dataset",
        min_traces=1,
        canonicalize_final_answer=True,
    )

    examples = [json.loads(line) for line in (tmp_path / "dataset" / "train.json").read_text().splitlines()]
    coordinator = next(example for example in examples if example["metadata"]["role"] == "coordinator")
    assert manifest["canonicalized_trace_ids"] == ["repairable"]
    assert coordinator["metadata"]["final_answer_canonicalized"]
    assert coordinator["messages"][-1]["content"] == '{"answer": 1}'
