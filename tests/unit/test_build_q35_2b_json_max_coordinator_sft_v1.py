from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_q35_2b_json_max_coordinator_sft_v1.py"
SPEC = importlib.util.spec_from_file_location("json_max_coordinator_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def episode() -> dict:
    return {
        "episode_id": "train_gen-json_max_two_shard-00000001-test",
        "split": "train_gen",
        "index": 1,
        "seed": 7,
        "public": {
            "system_prompt": builder.REQUIRED_SYSTEM_PROMPT,
            "user_prompt": (
                "review-worker owns /workspace/review.json. The coordinator owns "
                "/workspace/coordinator.json. Return remote_max, local_max, and global_max."
            ),
            "workspace_files": {"/workspace/coordinator.json": '{"a":-4,"b":9}'},
        },
        "oracle": {
            "children": [
                {
                    "name": "review-worker",
                    "resource_path": "/workspace/review.json",
                    "operation": "return the largest JSON integer value",
                    "expected_result": 6,
                }
            ],
            "resource_ownership": {
                "/workspace/review.json": {"owner": "child:review-worker", "family": "json_max"},
                "/workspace/coordinator.json": {"owner": "coordinator", "family": "json_max"},
            },
            "private_resources": {"/workspace/review.json": '{"x":6,"y":-2}'},
            "final_answer": {"remote_max": 6, "local_max": 9, "global_max": 9},
        },
    }


def runtime() -> dict:
    return {"role": "user", "content": "exact root runtime"}


def test_direct_row_parses_both_owned_shards_and_finalizes() -> None:
    direct = episode()
    direct["public"]["workspace_files"]["/workspace/review.json"] = direct["oracle"][
        "private_resources"
    ]["/workspace/review.json"]
    row = builder.direct_row(direct, runtime=runtime(), tools="[]")
    builder.validate_target_row(row)
    call = row["messages"][2]["tool_calls"][0]
    code = json.loads(call["function"]["arguments"])["code"]
    assert code.count("Path(") == 2
    assert "await rlm(" not in code
    assert row["messages"][-1]["content"] == '{"remote_max":6,"local_max":9,"global_max":9}'


def test_composed_rows_cover_spawn_wait_and_integration() -> None:
    rows = builder.composed_rows(episode(), runtime=runtime(), tools="[]")
    assert [row["phase"] for row in rows] == [
        "composed_spawn",
        "composed_wait",
        "composed_complete",
    ]
    for row in rows:
        builder.validate_target_row(row)
        call = row["messages"][2]["tool_calls"][0]
        code = json.loads(call["function"]["arguments"])["code"]
        assert code.count("await rlm(") == 1
        assert code.index("await rlm(") < code.index("Path(")
        assert "Path('/workspace/review.json')" not in code
    complete = rows[-1]
    assert complete["messages"][-2]["content"].endswith("\n\n6")
    assert complete["messages"][-1]["content"] == '{"remote_max":6,"local_max":9,"global_max":9}'


def test_runtime_contract_names_native_interfaces_without_conflict(tmp_path: Path) -> None:
    content = "\n".join(
        (
            "Recursive agent depth: 0",
            "await rlm('sub-task') returns immediately after task admission",
            "Children reply explicitly with `await agent_message.send",
            "Spawn independent children in separate calls and end your turn instead of awaiting completion.",
            builder.REQUIRED_SYSTEM_PROMPT,
        )
    )
    trace = {
        "traces": [
            {
                "nodes": [
                    {
                        "parent": None,
                        "sampled": False,
                        "message": {"role": "user", "content": content},
                    }
                ],
                "tools": [{"name": "ipython", "description": "run code", "parameters": {}}],
                "task": {"data": {"system_prompt": builder.REQUIRED_SYSTEM_PROMPT}},
            }
        ]
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(trace) + "\n")
    _runtime, tools, audit = builder.load_runtime_surface(path)
    assert [tool["name"] for tool in json.loads(tools)] == ["ipython"]
    assert audit["conflict_found"] is False


def test_target_validator_rejects_agent_message_as_spawn() -> None:
    row = builder.composed_rows(episode(), runtime=runtime(), tools="[]")[0]
    call = row["messages"][2]["tool_calls"][0]
    arguments = json.loads(call["function"]["arguments"])
    arguments["code"] = arguments["code"].replace("await rlm(", "await agent_message.send(")
    call["function"]["arguments"] = json.dumps(arguments)
    try:
        builder.validate_target_row(row)
    except ValueError:
        pass
    else:
        raise AssertionError("non-native spawn target was accepted")
