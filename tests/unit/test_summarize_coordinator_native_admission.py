import json
from pathlib import Path

from scripts.summarize_coordinator_native_admission import summarize


def _write_traces(path: Path, successes: dict[str, int]) -> None:
    rows = []
    for family, success_count in successes.items():
        for index in range(3):
            success = float(index < success_count)
            rows.append(
                {
                    "id": f"outer-{family}-{index}",
                    "traces": [
                        {
                            "id": f"trace-{family}-{index}",
                            "task": {"data": {"resource_family": family}},
                            "metrics": {
                                "strict_success": success,
                                "state_retained": success,
                            },
                        }
                    ],
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_passes_frozen_family_diversity_gate(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, {f"family-{index}": 2 if index < 2 else 1 for index in range(6)})

    report = summarize(traces)

    assert report["admission_pass"] is True
    assert report["strict_successes"] == 8
    assert len(report["success_families"]) == 6
    assert len(report["multi_success_families"]) == 2


def test_rejects_successes_without_family_diversity(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces, {"family-a": 3, "family-b": 3})

    report = summarize(traces)

    assert report["admission_pass"] is False
