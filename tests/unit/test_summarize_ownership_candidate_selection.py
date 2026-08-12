import json
from pathlib import Path

from scripts.summarize_ownership_candidate_selection import summarize


def _trace(index: int, success: bool, path_access: float = 0.0, spawn: float = 0.0) -> dict:
    return {
        "id": f"trace-{index}",
        "task": {
            "data": {
                "resource_family": f"family-{index}",
                "phrasing_variant": index % 2 + 2,
                "resource_path": f"/workspace/i{index}.json",
                "state_value": f"state-{index}",
            }
        },
        "metrics": {
            "strict_success": float(success),
            "parent_path_access": path_access,
            "one_spawn": spawn,
        },
    }


def _write(path: Path, traces: list[dict]) -> None:
    path.write_text("".join(json.dumps({"traces": [trace]}) + "\n" for trace in traces))


def test_selection_passes_only_with_six_gains_and_clean_restraint(tmp_path: Path) -> None:
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("cb", "cc", "db", "dc")}
    _write(paths["cb"], [_trace(i, False) for i in range(8)])
    _write(paths["cc"], [_trace(i, i < 4) for i in range(8)])
    _write(paths["db"], [_trace(i + 8, i < 4) for i in range(8)])
    _write(paths["dc"], [_trace(i + 8, i < 6) for i in range(8)])

    report = summarize(paths["cb"], paths["cc"], paths["db"], paths["dc"])

    assert report["strict_gains"] == 6
    assert report["strict_losses"] == 0
    assert report["promotion_pass"] is True


def test_selection_rejects_path_leakage_and_direct_delegation(tmp_path: Path) -> None:
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("cb", "cc", "db", "dc")}
    _write(paths["cb"], [_trace(i, False) for i in range(8)])
    _write(paths["cc"], [_trace(i, True, path_access=float(i == 0)) for i in range(8)])
    _write(paths["db"], [_trace(i + 8, False) for i in range(8)])
    _write(paths["dc"], [_trace(i + 8, False, spawn=float(i == 0)) for i in range(8)])

    report = summarize(paths["cb"], paths["cc"], paths["db"], paths["dc"])

    assert report["strict_gains"] == 8
    assert report["candidate_child_path_accesses"] == 1
    assert report["candidate_direct_spawns"] == 1
    assert report["promotion_pass"] is False
