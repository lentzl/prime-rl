import json

import pytest

from scripts.sample_natural_yield_sdpo_teacher_replays_v1 import (
    TeacherSamplingFailure,
    _audit_states,
    _classify_completion,
)


def test_classify_completion_distinguishes_passive_yield_from_tool_control():
    passive = _classify_completion(
        "I have delegated the private check and will wait for its result.<|im_end|>", finish_reason="stop"
    )
    polling = _classify_completion(
        '<tool_call>{"name":"agent_message.list_messages"}</tool_call>', finish_reason="stop"
    )
    truncated = _classify_completion("I should continue thinking", finish_reason="length")

    assert passive["uses_tool_or_poll"] is False
    assert passive["complete_no_tool_yield"] is True
    assert passive["category"] == "valid_passive_yield"
    assert polling["uses_tool_or_poll"] is True
    assert "<tool_call>" in polling["matched_markers"]
    assert polling["complete_no_tool_yield"] is False
    assert polling["category"] == "forbidden_tool_action"
    assert truncated["uses_tool_or_poll"] is False
    assert truncated["complete_no_tool_yield"] is False


def test_classify_completion_rejects_premature_finalization_and_empty_text():
    premature = _classify_completion(
        '{"finding": 17, "result": 29}<|im_end|>', finish_reason="stop"
    )
    empty = _classify_completion("<|im_end|>", finish_reason="stop")

    assert premature["category"] == "premature_finalization"
    assert empty["category"] == "other_invalid_no_tool"


def test_classify_completion_scores_visible_response_not_hidden_reasoning():
    completion = _classify_completion(
        "<think>Maybe the result is 17, but I have no evidence.</think>\n\n"
        "Waiting for the reviewer report.<|im_end|>",
        finish_reason="stop",
    )

    assert completion["visible_response"] == "Waiting for the reviewer report."
    assert completion["category"] == "valid_passive_yield"


def test_audit_states_recovers_exact_unconditioned_and_conditioned_prefixes(tmp_path):
    export_dir = tmp_path / "token_exports" / "step_1"
    export_dir.mkdir(parents=True)
    records = []
    for index in range(2):
        records.append(
            {
                "rank": index,
                "export_sequence_idx": 0,
                "token_ids": [10 + index, 20, 21, 22],
                "sdpo_weights": [None, 1.0, 1.0, 1.0],
                "sdpo_teacher_replays": [
                    {
                        "prefix_ids": [30 + index, 31],
                        "completion_ids": [20, 21, 22],
                        "student_positions": [1, 2, 3],
                        "target_offsets": [0, 1, 2],
                    }
                ],
            }
        )
    (export_dir / "rank_0.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )

    states = _audit_states(tmp_path, 2)

    assert [state["unconditioned_prefix_ids"] for state in states] == [[10], [11]]
    assert [state["conditioned_prefix_ids"] for state in states] == [[30, 31], [31, 31]]


def test_audit_states_rejects_ambiguous_failed_completion(tmp_path):
    export_dir = tmp_path / "token_exports" / "step_1"
    export_dir.mkdir(parents=True)
    record = {
        "token_ids": [10, 20, 10, 20],
        "sdpo_weights": [1.0, 1.0, 1.0, 1.0],
        "sdpo_teacher_replays": [
            {
                "prefix_ids": [30],
                "completion_ids": [10, 20],
                "student_positions": [0, 1],
                "target_offsets": [0, 1],
            }
        ],
    }
    (export_dir / "rank_0.jsonl").write_text(json.dumps(record) + "\n")

    with pytest.raises(TeacherSamplingFailure, match="exactly once"):
        _audit_states(tmp_path, 1)
