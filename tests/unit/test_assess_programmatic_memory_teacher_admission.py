import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "assess_programmatic_memory_teacher_admission.py"
SPEC = importlib.util.spec_from_file_location("assess_programmatic_memory_teacher_admission", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arm(count: int, strict: float, other: float = 1.0) -> dict:
    return {
        "count": count,
        "clean_count": count,
        "means": {
            metric: strict if metric == "strict_success" else other
            for metric in MODULE.CORE_METRICS
        },
        "diagnostics": {"expected_value_present": strict},
    }


def test_admits_reliable_teacher_with_substantial_error_reduction() -> None:
    report = MODULE.assess(
        {
            "familiar_unconditioned": arm(50, 0.88),
            "familiar_conditioned": arm(50, 0.96),
            "ood_unconditioned": arm(16, 0.75),
            "ood_conditioned": arm(16, 0.875),
        }
    )

    assert report["admission_pass"] is True
    assert report["failures"] == []


def test_rejects_ceiling_teacher_without_conditioning_gain() -> None:
    report = MODULE.assess(
        {
            "familiar_unconditioned": arm(50, 1.0),
            "familiar_conditioned": arm(50, 1.0),
            "ood_unconditioned": arm(16, 1.0),
            "ood_conditioned": arm(16, 1.0),
        }
    )

    assert report["admission_pass"] is False
    assert any("lacks a preregistered substantial gain" in item for item in report["failures"])


def test_rejects_incomplete_or_behaviorally_regressed_arm() -> None:
    conditioned = arm(15, 0.9)
    conditioned["means"]["retrieval_decision"] = 0.7
    report = MODULE.assess(
        {
            "familiar_unconditioned": arm(50, 0.5),
            "familiar_conditioned": arm(50, 0.95),
            "ood_unconditioned": arm(16, 0.9),
            "ood_conditioned": conditioned,
        }
    )

    assert report["admission_pass"] is False
    assert any("expected 16 traces" in item for item in report["failures"])
    assert any("retrieval_decision" in item for item in report["failures"])


def test_incomplete_audit_reports_rejection_without_crashing() -> None:
    empty = arm(0, 0.0, 0.0)
    report = MODULE.assess(
        {
            "familiar_unconditioned": arm(2, 1.0),
            "familiar_conditioned": empty,
            "ood_unconditioned": empty,
            "ood_conditioned": empty,
        }
    )

    assert report["admission_pass"] is False
    assert report["comparison"]["conditioned_strict"] == 0.0


def test_expected_value_presence_is_distinct_from_exact_output_contract() -> None:
    trace = {
        "task": {"data": {"expected_answers": ["8"]}},
        "nodes": [
            {
                "sampled": True,
                "message": {
                    "role": "assistant",
                    "content": "The latest stable checkpoint is **step 8**.",
                },
            }
        ],
        "metrics": {"answer_correct": 0.0},
    }

    assert MODULE.expected_value_present(trace) == 1.0
    assert MODULE.summarize([trace])["diagnostics"]["expected_value_present"] == 1.0
