import importlib.util
import json
import sys
from pathlib import Path


def _module():
    scripts = Path(__file__).parents[2] / "scripts"
    for name in (
        "summarize_q35_2b_topology_replications_v1",
        "summarize_q35_2b_recursive_compute_pair_v1",
    ):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        assert spec is not None and spec.loader is not None
        dependency = importlib.util.module_from_spec(spec)
        sys.modules[name] = dependency
        spec.loader.exec_module(dependency)
    name = "summarize_q35_2b_recursive_depth3_pair_v1"
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _trace(task: str, *, p_boundary: bool = False, d3: bool = False, correct: bool = False) -> dict:
    metrics = {
        "answer_accuracy": int(correct),
        "protocol_aligned": int(d3),
        "clean_protocol_aligned": int(d3),
        "topology_hierarchical": 1,
        "topology_utility_aligned": 1,
        "fan_in_complete": int(d3),
        "coordination_spawn_calls": 6 if d3 else 3,
        "coordination_failed_spawn_calls": 0 if d3 else (3 if p_boundary else 0),
        "depth3_top_manager_spawns": 1,
        "depth3_subgroup_manager_spawns": 2 if p_boundary or d3 else 0,
        "depth3_leaf_spawns": 3 if d3 else 0,
        "depth3_graph_complete": int(d3),
        "maximum_exercised_coordination_depth": 3 if d3 else 0,
        "failed_cells": 0 if d3 else 2,
        "coordinator_delegated_path_accesses": 0,
        "topology_direct": 0,
        "topology_flat": 0,
    }
    return {
        "id": f"{task}-{p_boundary}-{d3}-{correct}",
        "task": {"data": {"name": task, "family": "document_utility_depth3", "prompt": "same", "system_prompt": "same", "files": {"a": "b"}, "answer": {"x": 1}, "child_paths": {"document-manager": "/tmp"}}},
        "metrics": metrics,
        "stop_condition": "agent_completed",
    }


def _write(path: Path, traces: list[dict]) -> None:
    path.write_text("".join(json.dumps({"traces": [trace]}) + "\n" for trace in traces))


def test_depth3_selector_is_answer_free_and_requires_p_boundary(tmp_path: Path) -> None:
    module = _module()
    p_path, pplus_path = tmp_path / "p.jsonl", tmp_path / "pplus.jsonl"
    p = []
    pplus = []
    for index in range(4):
        task = f"task-{index}"
        p.append(_trace(task, p_boundary=index != 3))
        pplus.extend(
            [
                _trace(task, d3=True, correct=False),
                _trace(task, d3=True, correct=True),
                _trace(task),
            ]
        )
    _write(p_path, p)
    _write(pplus_path, pplus)

    summary = module.summarize_pair(p_path, pplus_path, expected_tasks=4)

    assert summary["selector_uses_expected_answer"] is False
    assert summary["p_boundary_eligible_count"] == 3
    assert summary["p_plus_selected_hard_successes"] == 0
    assert summary["qualifying_structural_gap_count"] == 0
    assert summary["gap_floor"] == 4
    assert summary["gap_floor_relaxed"] is False
    assert summary["passed"] is False
    assert all(row["p_plus"]["selected_attempt"] == 1 for row in summary["tasks"])


def test_depth3_auditor_passes_four_clean_gaps(tmp_path: Path) -> None:
    module = _module()
    p_path, pplus_path = tmp_path / "p.jsonl", tmp_path / "pplus.jsonl"
    p = []
    pplus = []
    for index in range(4):
        task = f"task-{index}"
        p.append(_trace(task, p_boundary=True))
        pplus.extend([_trace(task), _trace(task, d3=True, correct=True), _trace(task)])
    _write(p_path, p)
    _write(pplus_path, pplus)

    summary = module.summarize_pair(p_path, pplus_path, expected_tasks=4)

    assert summary["qualifying_structural_gap_count"] == 4
    assert summary["passed"] is True
    assert all(row["p_plus"]["selected_attempt"] == 2 for row in summary["tasks"])
