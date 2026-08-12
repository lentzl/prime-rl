import json
from pathlib import Path

from scripts.summarize_native_success_admission import summarize


def _write(path: Path, counts: dict[str, tuple[int, int]]) -> None:
    records = []
    for family, (rollouts, successes) in counts.items():
        for index in range(rollouts):
            records.append(
                {
                    "traces": [
                        {
                            "id": f"{family}-{index}",
                            "task": {"data": {"resource_family": family}},
                            "metrics": {"strict_success": float(index < successes)},
                        }
                    ]
                }
            )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_admission_requires_broad_and_repeated_native_success(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    _write(traces, {f"family-{index}": (16, 2 if index < 2 else 1) for index in range(6)})

    report = summarize(traces, minimum_families=6, minimum_multi_success_families=2)

    assert report["rollouts"] == 96
    assert report["strict_successes"] == 8
    assert report["families_with_success"] == 6
    assert report["families_with_multiple_successes"] == 2
    assert report["admission_pass"] is True


def test_admission_rejects_success_concentrated_in_too_few_families(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    _write(traces, {f"family-{index}": (16, 4 if index < 2 else 0) for index in range(8)})

    report = summarize(traces, minimum_families=6, minimum_multi_success_families=2)

    assert report["strict_successes"] == 8
    assert report["families_with_success"] == 2
    assert report["admission_pass"] is False
