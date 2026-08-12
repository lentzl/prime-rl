import argparse
import json
import sys
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


def test_ownership_admission_requires_strict_success() -> None:
    assert bootstrap.admitted(trace(strict_success=1.0), "ownership")
    assert not bootstrap.admitted(trace(strict_success=0.0), "ownership")
    assert bootstrap.admitted(trace(strict_success=1.0, reasoning=False), "ownership")
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


def test_count_requirements_are_strict_and_report_missing_keys() -> None:
    requirements = [
        bootstrap.parse_count_requirement("family.parallel=4"),
        bootstrap.parse_count_requirement("ownership.child.admitted_traces=2"),
    ]

    assert bootstrap.unmet_count_requirements(
        {"family.parallel": 3},
        requirements,
    ) == [
        "family.parallel=3<4",
        "ownership.child.admitted_traces=0<2",
    ]


@pytest.mark.parametrize("value", ["missing-separator", "family.parallel=-1", "=2"])
def test_invalid_count_requirement_is_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        bootstrap.parse_count_requirement(value)


def test_audit_only_reports_missing_requirements_without_writing(monkeypatch, capsys, tmp_path) -> None:
    manifest = {"counts": {"family.parallel": 3}, "rows": 3}
    monkeypatch.setattr(bootstrap, "build", lambda sources: ([{"messages_json": "[]"}], manifest))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_prime_agent_teacher_bootstrap.py",
            "--communication-run",
            str(tmp_path / "run"),
            "--require-count",
            "family.parallel=4",
            "--audit-only",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        bootstrap.main()

    assert exc.value.code is True
    assert json.loads(capsys.readouterr().out)["missing_requirements"] == ["family.parallel=3<4"]
    assert not list(tmp_path.glob("*.parquet"))
