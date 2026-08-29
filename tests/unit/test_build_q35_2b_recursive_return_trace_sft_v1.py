import importlib.util
import json
import sys
from pathlib import Path

import pytest
from datasets import Dataset


def _module():
    path = Path(__file__).parents[2] / "scripts" / "build_q35_2b_recursive_return_trace_sft_v1.py"
    spec = importlib.util.spec_from_file_location("recursive_return_trace_sft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _episode(code: str, *, reward: float = 1.0) -> dict:
    return {
        "id": "episode-1",
        "traces": [
            {
                "id": "trace-1",
                "stop_condition": "user_closed",
                "rewards": {"harness_score": {"score": reward}},
                "metrics": {"child_action_completed": 1},
                "tools": [{"type": "function", "function": {"name": "ipython"}}],
                "task": {
                    "key": "task-1",
                    "data": {
                        "oracle": {
                            "children": [{"name": "beta-worker", "expected_result": 17}]
                        },
                        "generation_metadata": {"resource_families": ["json_sum"]},
                    },
                },
                "nodes": [
                    {"parent": None, "sampled": False, "message": {"role": "user", "content": "root"}},
                    {"parent": 0, "sampled": True, "message": {"role": "assistant", "content": "root answer"}},
                    {"parent": None, "sampled": False, "message": {"role": "user", "content": "delegated"}},
                    {
                        "parent": 2,
                        "sampled": True,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "name": "ipython",
                                    "arguments": json.dumps({"code": code}),
                                }
                            ],
                        },
                    },
                    {
                        "parent": 3,
                        "sampled": False,
                        "message": {"role": "tool", "content": "queued", "tool_call_id": "call-1"},
                    },
                    {"parent": 4, "sampled": True, "message": {"role": "assistant", "content": "17"}},
                ],
            }
        ],
    }


def test_recursive_return_row_keeps_exact_action_through_ack_only() -> None:
    module = _module()
    code = "await agent_message.send('17', receiver_role='parent')"

    row = module.recursive_return_row(_episode(code))

    assert [message["role"] for message in row["messages"]] == ["user", "assistant", "tool"]
    assert json.loads(row["messages"][1]["tool_calls"][0]["arguments"]) == {"code": code}
    assert row["role"] == "coordinator_nonroot"
    assert row["resource_family"] == "json_sum"


def test_recursive_return_row_rejects_nonexact_or_nonqualifying_trace() -> None:
    module = _module()
    wrong = "await agent_message.send('18', receiver_role='parent')"

    try:
        module.recursive_return_row(_episode(wrong))
    except ValueError as exc:
        assert "not exact" in str(exc)
    else:
        raise AssertionError("nonexact return was accepted")

    try:
        module.recursive_return_row(_episode("await agent_message.send('17', receiver_role='parent')", reward=0.75))
    except ValueError as exc:
        assert "not a hard success" in str(exc)
    else:
        raise AssertionError("nonqualifying trace was accepted")


def test_qualifying_episode_filter_is_role_scoped_and_strict() -> None:
    module = _module()
    code = "await agent_message.send('17', receiver_role='parent')"
    passing = _episode(code)

    assert module.is_qualifying_episode(passing) is True
    passing["traces"][0]["metrics"]["child_action_completed"] = 0
    assert module.is_qualifying_episode(passing) is False


def test_child_only_natural_row_accepts_computed_delivery_despite_root_failure() -> None:
    module = _module()
    code = (
        "result = sum([39, 48, 63, 66, 33, 41])\n"
        "await agent_message.send(str(result), receiver_role='parent')"
    )
    episode = _episode(code, reward=0.75)

    assert module.is_qualifying_episode(
        episode, require_hard_success=False
    ) is True
    row = module.recursive_return_row(
        episode, allow_natural_action=True, require_hard_success=False
    )
    assert row["axis"] == "recursive_coordinator_return_natural_action"
    assert json.loads(row["messages"][1]["tool_calls"][0]["arguments"])["code"] == code

    wrong_route = _episode(
        "await agent_message.send('17', receiver_role='child')", reward=0.75
    )
    with pytest.raises(ValueError, match="natural parent send"):
        module.recursive_return_row(
            wrong_route, allow_natural_action=True, require_hard_success=False
        )


def test_child_only_cli_rejects_root_anchors(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    traces = tmp_path / "traces.jsonl"
    traces.write_text("", encoding="utf-8")
    anchors = tmp_path / "anchors.jsonl"
    anchors.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--forced-return-traces",
            str(traces),
            "--root-anchor-traces",
            str(anchors),
            "--child-only",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    with pytest.raises(ValueError, match="must not include root anchors"):
        module.main()


def test_replay_anchor_is_verified_repeated_and_interleaved(tmp_path: Path) -> None:
    module = _module()
    corpus = tmp_path / "anchor"
    corpus.mkdir()
    parquet = corpus / "train.parquet"
    Dataset.from_list(
        [
            {"role": "coordinator_nonroot", "trace_id": "old-1"},
            {"role": "coordinator_nonroot", "trace_id": "old-2"},
        ]
    ).to_parquet(str(parquet))
    (corpus / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": module.SCHEMA_VERSION,
                "status": "complete",
                "objective": "forced_recursive_child_return_consolidation",
                "root_retention_rows": 0,
                "dataset": {"sha256": module.sha256_file(parquet)},
            }
        ),
        encoding="utf-8",
    )

    anchors, source = module.replay_anchor_rows(corpus, repeats=2)

    assert [row["trace_id"] for row in anchors] == ["old-1", "old-2"] * 2
    assert source["parquet_sha256"] == module.sha256_file(parquet)
    assert [row["trace_id"] for row in module.interleave_rows(
        [{"trace_id": "new-1"}, {"trace_id": "new-2"}], anchors
    )[:4]] == ["new-1", "old-1", "new-2", "old-2"]
