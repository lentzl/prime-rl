import importlib.util
import json
import sys
from pathlib import Path


def _module():
    scripts = Path(__file__).parents[2] / "scripts"
    dependency_path = scripts / "summarize_q35_2b_topology_replications_v1.py"
    dependency_spec = importlib.util.spec_from_file_location(
        dependency_path.stem, dependency_path
    )
    assert dependency_spec is not None and dependency_spec.loader is not None
    dependency = importlib.util.module_from_spec(dependency_spec)
    sys.modules[dependency_spec.name] = dependency
    dependency_spec.loader.exec_module(dependency)

    path = scripts / "summarize_q35_2b_recursive_compute_pair_v1.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trace(task: str, *, structural: bool, correct: bool, trace_id: str) -> dict:
    metrics = {
        "answer_accuracy": int(correct),
        "protocol_aligned": int(structural),
        "clean_protocol_aligned": int(structural),
        "topology_valid": int(structural),
        "topology_utility_aligned": int(structural),
        "fan_in_complete": int(structural),
        "coordinator_delegated_path_accesses": 0,
        "failed_cells": 0,
        "duplicate_cells": 0,
        "post_parent_send_tool_calls": 0,
        "topology_direct": 0,
        "topology_flat": 0,
        "topology_hierarchical": int(structural),
        "coordination_spawn_calls": 4,
    }
    return {
        "id": trace_id,
        "task": {
            "data": {
                "name": task,
                "family": "document_utility_hierarchical",
                "prompt": "same prompt",
                "system_prompt": "same system",
                "files": {"/tmp/a": "evidence"},
                "answer": {"value": 7},
                "child_paths": {"document-manager": "/tmp"},
            }
        },
        "metrics": metrics,
        "stop_condition": "agent_completed" if structural else "max_turns",
    }


def _write(path: Path, traces: list[dict]) -> None:
    path.write_text(json.dumps({"traces": traces}) + "\n", encoding="utf-8")


def test_selector_is_answer_free_and_gap_floor_is_not_relaxed(tmp_path: Path) -> None:
    module = _module()
    p_path = tmp_path / "p.jsonl"
    pplus_path = tmp_path / "pplus.jsonl"
    p_traces = []
    pplus_traces = []
    for index in range(4):
        task = f"task-{index}"
        p_traces.append(
            _trace(task, structural=False, correct=False, trace_id=f"p-{index}")
        )
        pplus_traces.extend(
            [
                _trace(
                    task,
                    structural=True,
                    correct=False,
                    trace_id=f"pplus-{index}-wrong",
                ),
                _trace(
                    task,
                    structural=True,
                    correct=True,
                    trace_id=f"pplus-{index}-right",
                ),
                _trace(
                    task,
                    structural=False,
                    correct=False,
                    trace_id=f"pplus-{index}-failed",
                ),
            ]
        )
    _write(p_path, p_traces)
    _write(pplus_path, pplus_traces)

    summary = module.summarize_pair(
        p_path, pplus_path, expected_tasks=4, gap_floor=4
    )

    assert summary["selector_uses_expected_answer"] is False
    assert summary["p_plus_selected_hard_successes"] == 0
    assert summary["compute_gap_count"] == 0
    assert summary["gap_floor"] == 4
    assert summary["gap_floor_relaxed"] is False
    assert summary["passed"] is False
    assert all(row["p_plus"]["selected_attempt"] == 1 for row in summary["tasks"])


def test_reports_four_distinct_clean_compute_gaps(tmp_path: Path) -> None:
    module = _module()
    p_path = tmp_path / "p.jsonl"
    pplus_path = tmp_path / "pplus.jsonl"
    p_traces = []
    pplus_traces = []
    for index in range(4):
        task = f"task-{index}"
        p_traces.append(
            _trace(task, structural=False, correct=False, trace_id=f"p-{index}")
        )
        pplus_traces.extend(
            [
                _trace(
                    task,
                    structural=False,
                    correct=False,
                    trace_id=f"pplus-{index}-failed",
                ),
                _trace(
                    task,
                    structural=True,
                    correct=True,
                    trace_id=f"pplus-{index}-clean",
                ),
                _trace(
                    task,
                    structural=True,
                    correct=True,
                    trace_id=f"pplus-{index}-unused",
                ),
            ]
        )
    _write(p_path, p_traces)
    _write(pplus_path, pplus_traces)

    summary = module.summarize_pair(
        p_path, pplus_path, expected_tasks=4, gap_floor=4
    )

    assert summary["compute_gap_count"] == 4
    assert summary["passed"] is True
    assert all(row["p_plus"]["selected_attempt"] == 2 for row in summary["tasks"])
