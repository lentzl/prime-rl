from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")
pytest.importorskip("verifiers")

from scripts import export_prime_agent_teacher_bootstrap as bootstrap


def trace(*, ok=True, truncated=False, reasoning=True, **metrics):
    message = SimpleNamespace(
        role="assistant",
        reasoning_content="authentic sampled reasoning" if reasoning else None,
    )
    return SimpleNamespace(
        ok=ok,
        is_truncated=truncated,
        metrics=metrics,
        nodes=[SimpleNamespace(sampled=True, message=message)],
    )


def test_ownership_admission_requires_strict_success_and_sampled_reasoning() -> None:
    assert bootstrap.admitted(trace(strict_success=1.0), "ownership")
    assert not bootstrap.admitted(trace(strict_success=0.0), "ownership")
    assert not bootstrap.admitted(trace(strict_success=1.0, reasoning=False), "ownership")
    assert not bootstrap.admitted(trace(strict_success=1.0, truncated=True), "ownership")


def test_communication_admission_requires_answer_and_clean_protocol() -> None:
    assert bootstrap.admitted(
        trace(answer_accuracy=1.0, clean_protocol_aligned=1.0),
        "communication",
    )
    assert not bootstrap.admitted(
        trace(answer_accuracy=1.0, clean_protocol_aligned=0.0),
        "communication",
    )
    assert not bootstrap.admitted(
        trace(answer_accuracy=0.0, clean_protocol_aligned=1.0),
        "communication",
    )


def test_ownership_selects_only_the_root_coordinator_branch() -> None:
    coordinator_root = object()
    child_root = object()
    coordinator = SimpleNamespace(nodes=[coordinator_root])
    child = SimpleNamespace(nodes=[child_root])
    value = SimpleNamespace(
        nodes=[coordinator_root, child_root],
        branches=[child, coordinator],
    )

    assert bootstrap.coordinator_branches(value) == [coordinator]
