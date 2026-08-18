import json
from pathlib import Path

import pytest

from scripts.compare_natural_yield_sdpo_gates_v1 import GATES, compare


def _write_gate(
    root: Path,
    label: str,
    suffix: str,
    family: str,
    passed: int,
    *,
    exact: float = 1.0,
    local_work_before_yield: float = 1.0,
    premature_yield_before_local_work: float = 0.0,
    task_index: int = 1,
) -> None:
    gate = root / f"{label}-{suffix}" / "train-admission"
    gate.mkdir(parents=True)
    summary = {
        "rescored": True,
        "episodes": 8,
        "errors": 0,
        "harness": {"episodes": 8, "passed": passed, "rate": passed / 8},
        "diagnostic_means": {
            "final_answer_exact": exact,
            "local_work_before_yield": local_work_before_yield,
            "premature_yield_before_local_work": premature_yield_before_local_work,
        },
    }
    (gate / "SUMMARY.json").write_text(json.dumps(summary))
    task = {
        "episode_id": f"train_gen-{family}-{task_index:08d}",
        "family": family,
        "prompt": f"perform {family}",
        "system_prompt": "use the harness",
        "oracle": {"expected_route": family},
    }
    traces = [{"task": {"data": task}} for _ in range(8)]
    (gate / "traces.jsonl").write_text(json.dumps({"traces": traces}) + "\n")


def _write_battery(
    root: Path,
    label: str,
    scores: dict[str, int],
    *,
    exact: float = 1.0,
    changed_gate: str | None = None,
) -> None:
    families = {
        "natural_yield": "natural_n1",
        "natural_yield_local_work": "natural_n1",
        "atomic_state": "atomic_state",
        "atomic_send": "atomic_send",
    }
    for name, suffix in GATES.items():
        _write_gate(
            root,
            label,
            suffix,
            families[name],
            scores[name],
            exact=exact if name == "natural_yield" else 1.0,
            task_index=2 if name == changed_gate else 1,
        )


def test_comparison_authorizes_replication_only_after_target_gain_and_retention(
    tmp_path: Path,
) -> None:
    _write_battery(
        tmp_path,
        "r7",
        {
            "natural_yield": 0,
            "natural_yield_local_work": 2,
            "atomic_state": 8,
            "atomic_send": 5,
        },
    )
    _write_battery(
        tmp_path,
        "candidate",
        {
            "natural_yield": 2,
            "natural_yield_local_work": 2,
            "atomic_state": 8,
            "atomic_send": 6,
        },
    )

    report = compare(tmp_path, "r7", "candidate")

    assert report["decision"]["eligible_for_independent_replication"] is True
    assert report["decision"]["promoted"] is False
    assert report["gates"]["natural_yield"]["delta_passed"] == 2


def test_comparison_rejects_prerequisite_or_exact_answer_regression(tmp_path: Path) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(
        tmp_path,
        "candidate",
        {
            "natural_yield": 2,
            "natural_yield_local_work": 2,
            "atomic_state": 8,
            "atomic_send": 4,
        },
        exact=0.75,
    )

    report = compare(tmp_path, "r7", "candidate")

    assert report["decision"]["target_improved"] is True
    assert report["decision"]["prerequisites_retained"] is False
    assert report["decision"]["target_exact_answer_not_regressed"] is False
    assert report["decision"]["eligible_for_independent_replication"] is False


def test_comparison_fails_closed_on_different_task_draw(tmp_path: Path) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(tmp_path, "candidate", scores, changed_gate="natural_yield")

    with pytest.raises(ValueError, match="task specifications differ"):
        compare(tmp_path, "r7", "candidate")


def test_comparison_rejects_premature_yield_regression(tmp_path: Path) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(tmp_path, "candidate", scores)
    gate = (
        tmp_path
        / "candidate-natural-yield-local-work"
        / "train-admission"
        / "SUMMARY.json"
    )
    summary = json.loads(gate.read_text())
    summary["diagnostic_means"]["premature_yield_before_local_work"] = 0.25
    gate.write_text(json.dumps(summary))

    report = compare(tmp_path, "r7", "candidate")

    assert report["decision"]["anti_overgeneralization_retained"] is False
    assert report["decision"]["eligible_for_independent_replication"] is False
