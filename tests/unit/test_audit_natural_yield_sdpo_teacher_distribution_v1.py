import json
import math

import pytest

from scripts.audit_natural_yield_sdpo_teacher_distribution_v1 import (
    DistributionAuditFailure,
    _bounded_log_odds,
    _decision_record,
    _validate_state_diversity,
)


class FakeTokenizer:
    def convert_ids_to_tokens(self, token_id):
        return f"token-{token_id}"

    def decode(self, token_ids, skip_special_tokens=False):
        assert not skip_special_tokens
        return f"text-{token_ids[0]}"


def _support(token_ids, student_probs, teacher_probs):
    return {
        "token_ids": token_ids,
        "student_logprobs": [math.log(value) for value in student_probs],
        "teacher_logprobs": [math.log(value) for value in teacher_probs],
    }


def _record():
    return {
        "export_sequence_idx": 3,
        "loss_mask": [True, True, False],
        "sdpo_weights": [1.0, 1.0, 0.0],
        "sdpo_support": [
            {
                "position": 0,
                "student_support": _support([5, 6], [0.7, 0.2], [0.6, 0.2]),
                "teacher_support": _support([5, 6], [0.7, 0.2], [0.6, 0.2]),
            },
            {
                "position": 1,
                "student_support": _support([42, 7], [0.8, 0.1], [0.1, 0.2]),
                "teacher_support": _support([99, 7], [0.01, 0.1], [0.7, 0.2]),
            },
        ],
        "sdpo_teacher_replays": [
            {
                "prefix_ids": [1, 2],
                "completion_ids": [5, 42],
                "student_positions": [0, 1],
                "target_offsets": [0, 1],
            }
        ],
    }


def test_decision_record_reports_teacher_signal_outside_student_support():
    decision = _decision_record(
        _record(),
        tokenizer=FakeTokenizer(),
        topk=2,
        tool_marker_ids=[42],
        yield_token_id=99,
    )

    assert decision["decision_position"] == 1
    assert decision["student_top"]["id"] == 42
    assert decision["teacher_top"]["id"] == 99
    assert decision["teacher_top_is_tool"] is False
    assert decision["teacher_top_in_student_support"] is False
    assert decision["teacher_to_student_tool_ratio"] == pytest.approx(0.125)
    assert decision["yield_token_in_student_support"] is False
    assert decision["teacher_yield_probability"] == pytest.approx(0.7)


def test_decision_record_rejects_missing_tool_marker():
    record = _record()
    record["sdpo_teacher_replays"][0]["completion_ids"] = [5, 6]

    with pytest.raises(DistributionAuditFailure, match="does not contain"):
        _decision_record(
            record,
            tokenizer=FakeTokenizer(),
            topk=2,
            tool_marker_ids=[42],
            yield_token_id=99,
        )


@pytest.mark.parametrize(
    ("probability", "expected_sign"),
    [(0.0, -1), (1.0, 1)],
)
def test_bounded_log_odds_is_finite_at_probability_boundaries(probability, expected_sign):
    value = _bounded_log_odds(probability)

    assert math.isfinite(value)
    assert math.copysign(1, value) == expected_sign


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.inf, math.nan])
def test_bounded_log_odds_rejects_invalid_probability(probability):
    with pytest.raises(DistributionAuditFailure, match="invalid probability"):
        _bounded_log_odds(probability)


def test_state_diversity_requires_distinct_semantics_and_phrasings(tmp_path):
    path = tmp_path / "rollouts" / "step_1" / "train" / "effective"
    path.mkdir(parents=True)
    records = [
        {
            "task": {
                "data": {
                    "episode_id": f"episode-{index}",
                    "generation_metadata": {
                        "semantic_family": f"family-{index % 2}",
                        "control_contract_variant": f"style-{index % 3}",
                    },
                }
            }
        }
        for index in range(8)
    ]
    (path / "traces.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )

    report = _validate_state_diversity(tmp_path, 8)

    assert report["distinct_states"] == 8
    assert len(report["semantic_families"]) == 2
    assert len(report["phrasing_variants"]) == 3
