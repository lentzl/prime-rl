import json
from pathlib import Path

import verifiers.v1 as vf
from subagent_communication_v1.taskset import OWNERSHIP_GUIDANCE
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, SystemMessage, ToolCall, UserMessage

from scripts.export_ownership_teacher_sft import export


def _raw_trace(trace_id: str, task_name: str, *, valid: bool = True) -> dict:
    secret = 37
    path = f"/workspace/{task_name}.json"
    prompt = f"Read {path}, request the multiplier, then reply to the parent."
    code = f"multiplier = {secret}\nchild = await rlm(prompt={prompt!r}, name='key-worker')"
    if not valid:
        code = f"child = await rlm(prompt={prompt!r}, name='key-worker')"
    call = ToolCall(id="call-1", name="ipython", arguments=json.dumps({"code": code}))
    nodes = [
        MessageNode(
            parent=None,
            message=SystemMessage(content=f"Prime Agent system\n\n{OWNERSHIP_GUIDANCE}"),
            sampled=False,
        ),
        MessageNode(parent=0, message=UserMessage(content="Delegate this task."), sampled=False),
        MessageNode(
            parent=1,
            message=AssistantMessage(
                content=None,
                reasoning_content="I should preserve coordinator state and delegate the file.",
                tool_calls=[call],
            ),
            sampled=True,
        ),
    ]
    trace = vf.Trace(
        id=trace_id,
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, name=task_name)),
        nodes=nodes,
    ).model_dump(mode="json", exclude_none=True)
    trace["ok"] = True
    trace["task"]["data"].update(
        {
            "family": "followup",
            "expected_children": ["key-worker"],
            "child_paths": {"key-worker": path},
            "followup_secret": secret,
        }
    )
    trace["tools"] = [
        {
            "name": "ipython",
            "description": "Execute Python.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
        }
    ]
    return trace


def test_export_recomputes_gate_and_removes_guidance(tmp_path: Path) -> None:
    source = tmp_path / "traces.jsonl"
    source.write_text(
        json.dumps(
            {
                "traces": [
                    _raw_trace("accepted", "task-a"),
                    _raw_trace("rejected", "task-b", valid=False),
                ]
            }
        )
        + "\n"
    )

    manifest = export([source], tmp_path / "dataset", min_traces=1)
    example = json.loads((tmp_path / "dataset" / "train.json").read_text())

    assert manifest["accepted_trace_ids"] == ["accepted"]
    assert manifest["selection"]["ownership_transition"] == 1.0
    assert len(example["messages"]) == 3
    assert all(
        message.get("content") is None or isinstance(message["content"], str)
        for message in example["messages"]
    )
    assert example["messages"][-1]["reasoning_content"]
    assert example["messages"][-1]["tool_calls"][0]["function"]["name"] == "ipython"
    assert OWNERSHIP_GUIDANCE not in example["messages"][0]["content"]
    assert example["metadata"]["ownership_components"]["ownership_transition"] == 1.0


def test_export_caps_each_task_deterministically(tmp_path: Path) -> None:
    source = tmp_path / "traces.jsonl"
    source.write_text(
        json.dumps(
            {
                "traces": [
                    _raw_trace("first", "task-a"),
                    _raw_trace("second", "task-a"),
                    _raw_trace("third", "task-b"),
                ]
            }
        )
        + "\n"
    )

    manifest = export([source], tmp_path / "dataset", min_traces=2, max_per_task=1)

    assert manifest["accepted_before_task_cap"] == 3
    assert manifest["accepted_trace_ids"] == ["first", "third"]
    assert manifest["tasks"] == {"task-a": 1, "task-b": 1}
