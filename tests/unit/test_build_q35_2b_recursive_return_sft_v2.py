from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_q35_2b_recursive_return_sft_v2.py"
SPEC = importlib.util.spec_from_file_location("builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_computation_code_covers_each_family() -> None:
    cases = {
        "json_sum": ("[2, -1, 4]", "sum the top-level JSON integer list", 5),
        "csv_total": ("id,amount\n0,3\n1,8\n", "sum the CSV amount column", 11),
        "word_count": ("green ready green", "count exact 'green' tokens", 2),
        "md_h2": ("# A\n## B\nx\n## C\n", "count level-2 Markdown headings", 2),
        "log_error": ("INFO a\nERROR b\nERROR c\n", "count ERROR-level log lines", 2),
        "python_defs": (
            "def a():\n return 1\nasync def b():\n return 2",
            "count top-level sync and async function definitions",
            2,
        ),
        "json_max": (
            '{"m0": -2, "m1": 9}',
            "return the largest JSON integer value",
            9,
        ),
    }
    for family, (content, operation, expected) in cases.items():
        assert builder.compute_resource(family, content, operation) == expected
        code = builder.computation_code(family, content, operation)
        assert code.count("agent_message.send") == 1
        assert "receiver_role='parent'" in code


def test_recursive_row_has_one_action_and_no_trailing_assistant() -> None:
    episode = {
        "episode_id": "train_gen-natural_n1a-00000001-test",
        "oracle": {
            "children": [
                {
                    "name": "ledger-worker",
                    "resource_path": "/workspace/review.json",
                    "operation": "return the largest JSON integer value",
                    "expected_result": 8,
                }
            ],
            "resource_ownership": {
                "/workspace/review.json": {"family": "json_max"}
            },
            "private_resources": {"/workspace/review.json": '{"a": 3, "b": 8}'},
        },
    }
    row = builder.recursive_return_row(
        episode,
        template_system_message={"role": "user", "content": "runtime contract"},
        tools="[]",
    )
    assert [message["role"] for message in row["messages"]] == [
        "user", "user", "assistant", "tool"
    ]
    assert builder.RECURSIVE_HEADER in row["messages"][1]["content"]
    assert "evidence label is not a runtime path" in row["messages"][1]["content"]
    arguments = json.loads(row["messages"][2]["tool_calls"][0]["arguments"])
    assert arguments["code"].count("agent_message.send") == 1
    assert row["messages"][-1]["role"] == "tool"


def test_branch_paths_handles_a_fork() -> None:
    trace = {
        "nodes": [
            {"parent": None},
            {"parent": 0},
            {"parent": 1},
            {"parent": 1},
        ]
    }
    assert builder.branch_paths(trace) == [[0, 1, 2], [0, 1, 3]]
