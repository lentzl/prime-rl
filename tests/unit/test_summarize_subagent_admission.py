from __future__ import annotations

import json

from scripts.summarize_subagent_admission import summarize_file, summarize_trace

PATH = "/workspace/subagent-shards/v0-i5600-remote.json"


def _trace(*codes: str) -> dict:
    nodes = [
        {"parent": None, "message": {"role": "system", "content": "system"}},
        {"parent": 0, "message": {"role": "user", "content": "task"}},
    ]
    parent = 1
    for index, code in enumerate(codes):
        nodes.append(
            {
                "parent": parent,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "name": "ipython",
                            "arguments": json.dumps({"code": code}),
                        }
                    ],
                },
            }
        )
        parent = len(nodes) - 1
        nodes.append(
            {
                "parent": parent,
                "message": {
                    "role": "tool",
                    "tool_call_id": f"call-{index}",
                    "content": "",
                    "name": "ipython",
                },
            }
        )
        parent = len(nodes) - 1
    return {
        "task": {
            "data": {
                "family": "single",
                "name": "single-v0-i5600",
                "child_paths": {"shard-worker": PATH},
            }
        },
        "nodes": nodes,
    }


def test_exact_spawn_then_local_work_passes() -> None:
    prompt = (
        f"Read {PATH}, compute its checksum, then send it to your parent "
        "with agent_message."
    )
    result = summarize_trace(
        _trace(
            f"handle = await rlm({prompt!r}, name='shard-worker')",
            "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
        )
    )

    assert result["exact_admission"] is True
    assert result["two_stage_admission"] is True
    assert result["complete_admission"] is True
    assert result["parent_read_remote_first_cell"] is False


def test_same_cell_local_work_after_spawn_passes() -> None:
    prompt = (
        f"Read {PATH}, compute its checksum, then send it to your parent "
        "with agent_message."
    )
    result = summarize_trace(
        _trace(
            f"handle = await rlm({prompt!r}, name='shard-worker')\n"
            "local_values = [1, 2]\n"
            "local = sum(local_values)\n"
            "local"
        )
    )

    assert result["exact_admission"] is True
    assert result["separate_local_work"] is False
    assert result["local_work_after_spawn"] is True
    assert result["two_stage_admission"] is True
    assert result["complete_admission"] is True


def test_generic_spawn_payload_fails() -> None:
    result = summarize_trace(
        _trace(
            "handle = await rlm('sub-task', name='shard-worker')",
            "local_values = [1, 2]\nlocal = sum(local_values)\nlocal",
        )
    )

    assert result["retained_handle"] is True
    assert result["exact_payload"] is False
    assert result["two_stage_admission"] is False
    assert result["complete_admission"] is False


def test_parent_remote_read_before_spawn_fails() -> None:
    prompt = (
        f"Read {PATH}, compute its checksum, then send it to your parent "
        "with agent_message."
    )
    result = summarize_trace(
        _trace(
            f"remote = json.loads(Path({PATH!r}).read_text())\n"
            f"handle = await rlm({prompt!r}, name='shard-worker')"
        )
    )

    assert result["exact_payload"] is True
    assert result["parent_read_remote_first_cell"] is True
    assert result["exact_admission"] is False
    assert result["complete_admission"] is False


def test_local_work_before_spawn_is_not_complete_admission() -> None:
    prompt = (
        f"Read {PATH}, compute its checksum, then send it to your parent "
        "with agent_message."
    )
    result = summarize_trace(
        _trace(
            "local_values = [1, 2]\n"
            "local = sum(local_values)\n"
            f"handle = await rlm({prompt!r}, name='shard-worker')"
        )
    )

    assert result["exact_admission"] is True
    assert result["spawn_precedes_local"] is False
    assert result["complete_admission"] is False


def test_file_summary_skips_failed_empty_trace(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "task": {"data": {"family": "single"}},
                        "nodes": [],
                        "errors": [{"type": "ProviderError"}],
                    }
                ]
            }
        )
    )

    result = summarize_file(path)

    assert result["traces"] == 0
    assert result["skipped_traces"] == 1
