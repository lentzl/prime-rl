import pytest

from scripts.decide_natural_yield_sdpo_teacher_admission_v1 import (
    TeacherAdmissionFailure,
    decide,
)


def _distribution(*, shift_rate=0.875, mean_shift=0.4):
    return {
        "verdict": "pass",
        "model_artifacts_written": False,
        "state_diversity": {"distinct_states": 8},
        "distribution": {
            "summary": {
                "positive_away_from_tool_shift_rate": shift_rate,
                "mean_away_from_tool_log_odds_shift": mean_shift,
                "teacher_signal_present": shift_rate >= 0.75 and mean_shift > 0,
            }
        },
    }


def _behavior(*, conditioned=0.5, gain=0.25, distributed=5):
    admitted = conditioned >= 0.3 and gain >= 0.2 and distributed >= 4
    return {
        "verdict": "pass",
        "summary": {
            "states": 8,
            "yield_rates": {"unconditioned": conditioned - gain, "conditioned": conditioned},
            "conditioned_absolute_yield_gain": gain,
            "states_with_conditioned_yield_and_forbidden_tool_reduction": distributed,
            "behavioral_teacher_admitted": admitted,
        },
    }


def test_teacher_admission_requires_both_gates():
    report = decide(_distribution(), _behavior())

    assert report["behavioral_pass"] is True
    assert report["distributional_pass"] is True
    assert report["teacher_update_authorized"] is True


def test_teacher_admission_rejects_logits_without_behavioral_yield():
    report = decide(_distribution(), _behavior(conditioned=0.1, gain=0.1, distributed=1))

    assert report["distributional_pass"] is True
    assert report["behavioral_pass"] is False
    assert report["teacher_update_authorized"] is False
    assert report["next_action"] == "change_teacher_source_or_representation_without_gradient"


def test_teacher_admission_rejects_mismatched_state_counts():
    behavior = _behavior()
    behavior["summary"]["states"] = 7

    with pytest.raises(TeacherAdmissionFailure, match="share at least eight states"):
        decide(_distribution(), behavior)
