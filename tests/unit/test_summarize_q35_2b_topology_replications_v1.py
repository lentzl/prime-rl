import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[2]
        / "scripts"
        / "summarize_q35_2b_topology_replications_v1.py"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trace(family: str, selected: str, index: int) -> dict:
    expected = family.removeprefix("document_utility_")
    delegated = expected != "direct"
    metrics = {
        "answer_accuracy": 1,
        "protocol_aligned": int(selected == expected),
        "clean_protocol_aligned": int(selected == expected),
        "topology_valid": 1,
        "topology_utility_aligned": int(selected == expected),
        "coordinator_delegated_path_accesses": 0,
        "failed_cells": 0,
        "duplicate_cells": 0,
        "post_parent_send_tool_calls": 0,
        "fan_in_complete": int(delegated),
    }
    for topology in ("direct", "flat", "hierarchical"):
        metrics[f"topology_{topology}"] = int(selected == topology)
    return {
        "task": {"data": {"family": family, "name": f"{family}-{index}"}},
        "metrics": metrics,
        "stop_condition": "agent_completed",
    }


def _write_bank(path: Path, flat_selections: tuple[str, str]) -> None:
    traces = []
    for family, selections in (
        ("document_utility_direct", ("direct", "direct")),
        ("document_utility_flat", flat_selections),
        ("document_utility_hierarchical", ("hierarchical", "hierarchical")),
    ):
        traces.extend(_trace(family, selected, index) for index, selected in enumerate(selections))
    path.write_text(json.dumps({"traces": traces}) + "\n", encoding="utf-8")


def test_reports_confusion_and_preserves_four_of_six_floor(tmp_path: Path) -> None:
    module = _module()
    first = tmp_path / "v14.jsonl"
    second = tmp_path / "v15.jsonl"
    _write_bank(first, ("hierarchical", "hierarchical"))
    _write_bank(second, ("flat", "hierarchical"))

    summary = module.summarize_replications([("v14", first), ("v15", second)])

    assert summary["acceptance_floor"] == 4
    assert summary["acceptance_floor_relaxed"] is False
    assert summary["all_banks_passed"] is True
    assert [bank["qualifying_trajectories"] for bank in summary["banks"]] == [4, 5]
    assert summary["aggregate_confusion"]["flat"] == {
        "direct": 0,
        "flat": 1,
        "hierarchical": 3,
        "invalid": 0,
    }


def test_hard_invariant_failure_does_not_qualify(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "bank.jsonl"
    _write_bank(path, ("flat", "flat"))
    outer = json.loads(path.read_text())
    outer["traces"][0]["metrics"]["failed_cells"] = 1
    path.write_text(json.dumps(outer) + "\n", encoding="utf-8")

    bank = module.summarize_bank("bank", path)

    assert bank["qualifying_trajectories"] == 5
    assert bank["traces"][0]["reasons"] == ["failed_cells"]
