import json
from pathlib import Path

import pytest

from scripts.compare_procedural_harness_master_checkpoints_v1 import compare


def _write_summary(root: Path, label: str, split: str, passed: int, *, errors: int = 0) -> None:
    path = root / label / split
    path.mkdir(parents=True)
    report = {
        "rescored": True,
        "episodes": 24,
        "errors": errors,
        "harness": {"episodes": 24, "passed": passed, "rate": passed / 24},
        "by_family": {"single": {"episodes": 4, "passed": passed, "rate": passed / 24}},
        "diagnostic_means": {"required_atoms_fraction": 0.5},
    }
    (path / "SUMMARY.json").write_text(json.dumps(report))


def test_checkpoint_screen_requires_improvement_without_split_regression(tmp_path: Path) -> None:
    for split, passed in (("valid-baseline", 5), ("ood-baseline", 3)):
        _write_summary(tmp_path, "untouched", split, passed)
    for split, passed in (("valid-baseline", 5), ("ood-baseline", 5)):
        _write_summary(tmp_path, "step-1", split, passed)
    for split, passed in (("valid-baseline", 4), ("ood-baseline", 6)):
        _write_summary(tmp_path, "step-2", split, passed)

    report = compare(tmp_path, expected_steps=2)

    assert report["recommended_for_replication"] == "step-1"
    assert report["candidates"][1]["screen_pass"] is True
    assert report["candidates"][1]["split_deltas"] == {
        "valid-baseline": 0,
        "ood-baseline": 2,
    }
    assert report["candidates"][2]["screen_pass"] is False


def test_checkpoint_screen_fails_closed_on_incomplete_or_errorful_summary(tmp_path: Path) -> None:
    for split in ("valid-baseline", "ood-baseline"):
        _write_summary(tmp_path, "untouched", split, 1)
    _write_summary(tmp_path, "step-1", "valid-baseline", 2)
    _write_summary(tmp_path, "step-1", "ood-baseline", 2, errors=1)

    with pytest.raises(ValueError, match="rollout errors"):
        compare(tmp_path, expected_steps=1)
