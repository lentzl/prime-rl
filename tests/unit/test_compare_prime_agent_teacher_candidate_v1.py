import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "compare_prime_agent_teacher_candidate_v1.py"
SPEC = importlib.util.spec_from_file_location("compare_prime_agent_teacher_candidate_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _summary(count: int, issue_names: set[str] | None = None) -> dict:
    issue_names = issue_names or set()
    families = {}
    tasks = []
    for index in range(count):
        family = "direct" if index == 0 else "single"
        name = f"{family}-{index}"
        issues = ["protocol"] if name in issue_names else []
        tasks.append({"name": name, "family": family, "issues": issues})
        data = families.setdefault(
            family,
            {"count": 0, "clean_count": 0, "mean_reward": 1.0, "means": {}},
        )
        data["count"] += 1
        data["clean_count"] += not issues
    return {
        "trace_count": count,
        "issue_count": len(issue_names),
        "families": families,
        "tasks": tasks,
    }


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def _paths(tmp_path: Path, base_issues: set[str], candidate_issues: set[str]) -> tuple[Path, ...]:
    base_mastery = _summary(74, base_issues)
    candidate_mastery = _summary(74, candidate_issues)
    base_resilience = _summary(12)
    candidate_resilience = _summary(12)
    return (
        _write(tmp_path / "base-mastery.json", base_mastery),
        _write(tmp_path / "candidate-mastery.json", candidate_mastery),
        _write(tmp_path / "base-resilience.json", base_resilience),
        _write(tmp_path / "candidate-resilience.json", candidate_resilience),
    )


def test_comparison_admits_multiple_target_gains_without_losses(tmp_path: Path) -> None:
    paths = _paths(tmp_path, {"single-1", "single-2"}, set())

    report = MODULE.compare(*paths)

    assert report["verdict"] == "PROMOTION-ELIGIBLE-PENDING-INDEPENDENT-REPEAT"
    assert report["target_clean_gains"] == 2
    assert report["clean_losses"] == []


def test_comparison_rejects_a_protected_direct_regression(tmp_path: Path) -> None:
    paths = _paths(tmp_path, {"single-1", "single-2"}, {"direct-0"})

    report = MODULE.compare(*paths)

    assert report["verdict"] == "BRANCH-REJECTED"
    assert report["hard_rejections"] == ["foundation_or_direct_clean_loss"]


def test_comparison_requires_exact_task_identity(tmp_path: Path) -> None:
    paths = list(_paths(tmp_path, set(), set()))
    candidate = json.loads(paths[1].read_text())
    candidate["tasks"][1]["name"] = "different-task"
    paths[1].write_text(json.dumps(candidate))

    with pytest.raises(MODULE.ComparisonFailure, match="task identities differ"):
        MODULE.compare(*paths)
