import json
from pathlib import Path

from scripts.summarize_ownership_invariant_admission import summarize


def _trace(family: str, phrasing: int, success: bool) -> dict:
    return {
        "task": {
            "data": {
                "name": f"{family}-p{phrasing}",
                "resource_family": family,
                "phrasing_variant": phrasing,
            }
        },
        "metrics": {"strict_success": float(success)},
    }


def test_summary_applies_frozen_phase_a_gate(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    records = []
    for index, family in enumerate(("json", "csv", "text", "markdown")):
        traces = [
            _trace(family, index % 2, True),
            _trace(family, index % 2, False),
        ]
        records.append({"traces": traces})
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    result = summarize(path)

    assert result["phase_a_pass"] is True
    assert result["mixed_groups"] == 4
    assert len(result["success_resource_families"]) == 4
    assert result["success_phrasings"] == [0, 1]
