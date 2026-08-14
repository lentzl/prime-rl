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


def test_partial_conditioned_arm_reports_mathematical_rejection_ceiling() -> None:
    conditioned = arm(34, 20 / 34)
    report = MODULE.assess(
        {
            "familiar_unconditioned": arm(50, 0.2),
            "familiar_conditioned": conditioned,
            "ood_unconditioned": arm(0, 0.0, 0.0),
            "ood_conditioned": arm(0, 0.0, 0.0),
        }
    )

    assert report["admission_still_possible"] is False
    assert any(
        "familiar: conditioned strict_success can reach at most 0.720"
        in item
        for item in report["early_rejection_reasons"]
    )


def test_partial_conditioned_arm_remains_possible_above_threshold_ceiling() -> None:
    report = MODULE.assess(
        {
            "familiar_unconditioned": arm(50, 0.2),
            "familiar_conditioned": arm(34, 0.9),
            "ood_unconditioned": arm(16, 0.2),
            "ood_conditioned": arm(10, 0.8),
        }
    )

    assert report["admission_still_possible"] is True
    assert report["early_rejection_reasons"] == []


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


def test_task_identity_ignores_only_conditioned_system_prompt() -> None:
    base = {
        "task": {
            "data": {
                "name": "familiar_heldout-latest_state-9",
                "prompt": "Recover the value.",
                "system_prompt": "ordinary system prompt",
                "expected_answers": ["stable"],
                "files": {"/workspace/history.log": "event"},
            }
        }
    }
    conditioned = {
        "task": {
            "data": {
                **base["task"]["data"],
                "system_prompt": "expert demonstration followed by ordinary system prompt",
            }
        }
    }

    assert MODULE.task_identity(base) == MODULE.task_identity(conditioned)
    conditioned["task"]["data"]["expected_answers"] = ["beta"]
    assert MODULE.task_identity(base) != MODULE.task_identity(conditioned)


def test_rejects_unpaired_conditioned_tasks() -> None:
    arms = {
        "familiar_unconditioned": arm(50, 0.88),
        "familiar_conditioned": arm(50, 0.96),
        "ood_unconditioned": arm(16, 0.75),
        "ood_conditioned": arm(16, 0.875),
    }
    for value in arms.values():
        value["task_identity_counts"] = {"same": value["count"]}
    arms["familiar_conditioned"]["task_identity_counts"] = {"different": 50}

    report = MODULE.assess(arms)

    assert report["admission_pass"] is False
    assert any("same frozen task identities" in item for item in report["failures"])
