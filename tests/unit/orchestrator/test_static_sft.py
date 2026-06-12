import pytest

from prime_rl.configs.algorithm import DatasetConfig
from prime_rl.orchestrator.static_sft import load_static_sft_rows, static_sft_rollout


def test_static_sft_rollout_splits_messages_on_assistant_turns():
    row = {
        "example_id": "ex-1",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "Two"},
            {"role": "user", "content": "Three"},
            {"role": "assistant", "content": "Four"},
        ],
    }

    rollout = static_sft_rollout(row, DatasetConfig(name="org/static-sft"))

    assert rollout["example_id"] == "ex-1"
    assert rollout["error"] is None
    assert rollout["reward"] == 1.0
    assert rollout["stop_condition"] == "replayed_messages"
    assert len(rollout["trajectory"]) == 2
    assert rollout["trajectory"][0]["prompt"] == row["messages"][:2]
    assert rollout["trajectory"][0]["completion"] == [row["messages"][2]]
    assert rollout["trajectory"][0]["tokens"] is None
    assert rollout["trajectory"][1]["prompt"] == row["messages"][:4]
    assert rollout["trajectory"][1]["completion"] == [row["messages"][4]]


def test_static_sft_rollout_accepts_prompt_completion_columns():
    row = {"prompt": "Question", "completion": "Answer"}

    rollout = static_sft_rollout(row, DatasetConfig(name="org/static-sft"))

    assert rollout["trajectory"][0]["prompt"] == [{"role": "user", "content": "Question"}]
    assert rollout["trajectory"][0]["completion"] == [{"role": "assistant", "content": "Answer"}]


def test_static_sft_rollout_requires_assistant_target():
    with pytest.raises(ValueError, match="no assistant messages"):
        static_sft_rollout(
            {"messages": [{"role": "user", "content": "Question"}]},
            DatasetConfig(name="org/static-sft"),
        )


def test_static_sft_rollout_respects_max_turns():
    row = {
        "messages": [
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "Two"},
            {"role": "user", "content": "Three"},
            {"role": "assistant", "content": "Four"},
        ],
    }

    rollout = static_sft_rollout(row, DatasetConfig(name="org/static-sft", max_turns=1))

    assert rollout["stop_condition"] == "max_turns_reached"
    assert rollout["is_truncated"] is True
    assert len(rollout["trajectory"]) == 1
    assert rollout["trajectory"][0]["is_truncated"] is True


def test_load_static_sft_rows_reads_local_jsonl(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "examples.jsonl").write_text(
        '{"messages":[{"role":"user","content":"One"},{"role":"assistant","content":"Two"}]}\n'
        '{"example_id":"kept","messages":[{"role":"user","content":"Three"},{"role":"assistant","content":"Four"}]}\n'
    )

    rows = load_static_sft_rows(DatasetConfig(data_dir=data_dir))

    assert rows[0]["example_id"] == 0
    assert rows[1]["example_id"] == "kept"
