import json
from pathlib import Path

import pytest

from scripts.compare_procedural_harness_action_gates_v1 import REQUIRED_RUNGS, compare


def _write_gate(
    root: Path,
    label: str,
    rung: str,
    passed: int,
    *,
    task_index: int = 1,
    errors: int = 0,
) -> None:
    gate = root / f"{label}-{rung}-gate-r1" / "train-admission"
    gate.mkdir(parents=True)
    diagnostics = {
        "all_required_atoms": passed / 8,
        "required_atoms_fraction": 0.5 + passed / 16,
    }
    summary = {
        "rescored": True,
        "episodes": 8,
        "errors": errors,
        "harness": {"episodes": 8, "passed": passed, "rate": passed / 8},
        "by_family": {rung: {"episodes": 8, "passed": passed, "rate": passed / 8}},
        "diagnostic_means": diagnostics,
    }
    (gate / "SUMMARY.json").write_text(json.dumps(summary))
    task = {
        "episode_id": f"train_gen-{rung}-{task_index:08d}",
        "family": rung,
        "prompt": f"perform {rung}",
        "system_prompt": "use the harness",
        "oracle": {"expected_route": rung},
    }
    traces = [{"task": {"data": task}} for _ in range(8)]
    (gate / "traces.jsonl").write_text(json.dumps({"traces": traces}) + "\n")


def test_action_gate_comparison_requires_target_gain_and_retention(tmp_path: Path) -> None:
    base_scores = {
        "atomic_state": 8,
        "atomic_send": 6,
        "atomic_child_request": 5,
        "atomic_followup": 0,
    }
    candidate_scores = {
        "atomic_state": 8,
        "atomic_send": 7,
        "atomic_child_request": 8,
        "atomic_followup": 1,
    }
    for rung in REQUIRED_RUNGS:
        _write_gate(tmp_path, "base", rung, base_scores[rung])
        _write_gate(tmp_path, "candidate", rung, candidate_scores[rung])

    report = compare(tmp_path, "base", "candidate")

    assert report["screen"]["pass"] is True
    assert report["rungs"]["atomic_child_request"]["delta_passed"] == 3
    assert report["rungs"]["atomic_send"]["diagnostic_deltas"]["all_required_atoms"] == 0.125


def test_action_gate_comparison_fails_closed_on_mismatched_draw(tmp_path: Path) -> None:
    for rung in REQUIRED_RUNGS:
        _write_gate(tmp_path, "base", rung, 4)
        _write_gate(
            tmp_path,
            "candidate",
            rung,
            5,
            task_index=2 if rung == "atomic_child_request" else 1,
        )

    with pytest.raises(ValueError, match="different task specifications"):
        compare(tmp_path, "base", "candidate")


def test_action_gate_comparison_rejects_prerequisite_regression(tmp_path: Path) -> None:
    for rung in REQUIRED_RUNGS:
        _write_gate(tmp_path, "base", rung, 4)
        candidate_passed = 6 if rung == "atomic_child_request" else 4
        if rung == "atomic_send":
            candidate_passed = 3
        _write_gate(tmp_path, "candidate", rung, candidate_passed)

    report = compare(tmp_path, "base", "candidate")

    assert report["screen"]["target_improved"] is True
    assert report["screen"]["prerequisites_retained"] is False
    assert report["screen"]["pass"] is False
