import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[2] / "scripts" / "export_prime_agent_role_sft_v1.py"
    spec = importlib.util.spec_from_file_location("export_prime_agent_role_sft_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace():
    return {
        "id": "trace-1",
        "nodes": [
            {"parent": None, "sampled": False, "message": {"role": "user", "content": "root"}},
            {"parent": 0, "sampled": True, "message": {"role": "assistant", "content": "spawn"}},
            {"parent": 1, "sampled": False, "message": {"role": "user", "content": "child reply"}},
            {"parent": 2, "sampled": True, "message": {"role": "assistant", "content": "done"}},
            {"parent": None, "sampled": False, "message": {"role": "user", "content": "child"}},
            {"parent": 4, "sampled": True, "message": {"role": "assistant", "content": "report"}},
        ],
        "calls": [
            {"node": 1, "client_session_id": "root-session"},
            {"node": 3, "client_session_id": "root-session"},
            {"node": 5, "client_session_id": "child-session"},
        ],
        "tools": [],
    }


def test_role_rows_separate_root_and_child_sessions():
    module = _module()
    root_rows, root_sessions = module.role_rows(
        _trace(), episode_id="episode-1", axis="natural_n1a", role="orchestrator"
    )
    child_rows, child_sessions = module.role_rows(_trace(), episode_id="episode-1", axis="natural_n1a", role="child")

    assert root_sessions == ["root-session"]
    assert child_sessions == ["child-session"]
    assert [message["content"] for message in root_rows[0]["messages"]] == [
        "root",
        "spawn",
        "child reply",
        "done",
    ]
    assert [message["content"] for message in child_rows[0]["messages"]] == ["child", "report"]


def test_role_rows_fail_closed_on_missing_lineage():
    module = _module()
    trace = _trace()
    trace["calls"] = [call for call in trace["calls"] if call["node"] != 5]

    with pytest.raises(ValueError, match="ambiguous session lineage"):
        module.role_rows(trace, episode_id="episode-1", axis="natural_n1a", role="child")


def test_direct_control_has_no_child_training_row():
    module = _module()
    trace = _trace()
    trace["nodes"] = trace["nodes"][:4]
    trace["calls"] = trace["calls"][:2]

    rows, sessions = module.role_rows(
        trace,
        episode_id="episode-1",
        axis="natural_direct_control",
        role="child",
    )

    assert rows == []
    assert sessions == []
