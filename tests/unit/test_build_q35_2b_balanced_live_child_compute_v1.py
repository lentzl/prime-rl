from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[2] / "scripts" / "build_q35_2b_balanced_live_child_compute_v1.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("balanced_live_child_compute", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


def _episode() -> dict:
    return {
        "episode_id": "train_gen-natural_n1a-00000001-test",
        "index": 1,
        "oracle": {
            "children": [
                {
                    "name": "beta-worker",
                    "resource_path": "/workspace/review.txt",
                    "operation": "count exact 'green' tokens",
                    "expected_result": 2,
                }
            ],
            "resource_ownership": {
                "/workspace/review.txt": {"family": "word_count"}
            },
            "private_resources": {
                "/workspace/review.txt": "green ready green done"
            },
        },
    }


def test_live_generic_prefix_requires_current_exact_contract() -> None:
    module = _module()
    row = {
        "messages": [
            {"role": "system", "content": module.LEAF_REPORTER_CONTRACT},
            {
                "role": "user",
                "content": "generic runtime\n\n[recursive coordinator session contract]\ndynamic",
            },
            {"role": "user", "content": "task"},
        ]
    }

    assert module.live_generic_prefix(row) == "generic runtime"
    row["messages"][0]["content"] = "old contract"
    with pytest.raises(ValueError, match="current leaf reporter"):
        module.live_generic_prefix(row)


def test_balanced_live_row_is_exact_context_and_answer_free() -> None:
    module = _module()

    row = module.balanced_live_row(
        _episode(), generic_prefix="generic runtime", tools="[]"
    )

    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
    ]
    assert row["messages"][0]["content"] == module.LEAF_REPORTER_CONTRACT
    assert row["messages"][1]["content"].startswith("generic runtime\n\n")
    assert "Required review: count exact 'green' tokens" in row["messages"][1]["content"]
    assert "[task from parent]" in row["messages"][2]["content"]
    code = json.loads(row["messages"][3]["tool_calls"][0]["arguments"])["code"]
    assert "INLINE_EVIDENCE.split().count(keyword)" in code
    assert "green ready green done" not in code
    assert "receiver_role='parent'" in code
    assert row["expected_result"] == "2"
