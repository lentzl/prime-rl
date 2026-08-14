import json
from pathlib import Path

import pytest

from scripts.compare_memory_v2_tranche_qualification import compare
from scripts.summarize_prime_agent_mastery import summarize as summarize_mastery
from scripts.summarize_programmatic_memory_eval import (
    FEEDBACK_SCHEMA,
)
from scripts.summarize_programmatic_memory_eval import (
    report as report_memory,
)


def _memory_trace(*, idx: int, strict: float, family: str = "latest_state") -> dict:
    info = {}
    if strict < 1.0:
        contract = {
            "schema_version": FEEDBACK_SCHEMA,
            "code": "event_semantics_mismatch",
            "category": "event_semantics",
            "answer_free": True,
            "retryable": True,
            "message": "Use the latest valid event.",
        }
        info = {"feedback": contract["message"], "feedback_contract": contract}
    return {
        "ok": True,
        "task": {
            "data": {
                "idx": idx,
                "name": f"familiar_heldout-{family}-{idx}",
                "split": "familiar_heldout",
                "family": family,
            }
        },
        "metrics": {
            "strict_success": strict,
            "answer_correct": strict,
            "retrieval_decision": 1.0,
        },
        "info": info,
    }


def _mastery_trace(*, name: str, path_accesses: float) -> dict:
    return {
        "ok": True,
        "task": {"data": {"name": name, "family": "child"}},
        "metrics": {
            "answer_accuracy": 1.0,
            "strict_success": 1.0,
            "parent_path_access": path_accesses,
        },
        "info": {"env_name": "subagent-communication-v1"},
        "rewards": {"reward": {"score": 1.0}},
    }


def _write_model(root: Path, label: str, memory: list[dict], mastery: list[dict]) -> None:
    model_root = root / label
    memory_root = model_root / "memory" / "results"
    mastery_root = model_root / "mastery" / "results"
    memory_root.mkdir(parents=True)
    mastery_root.mkdir(parents=True)
    (memory_root / "traces.jsonl").write_text(
        "\n".join(json.dumps({"traces": [trace]}) for trace in memory) + "\n",
        encoding="utf-8",
    )
    (mastery_root / "traces.jsonl").write_text(
        "\n".join(json.dumps({"traces": [trace]}) for trace in mastery) + "\n",
        encoding="utf-8",
    )
    (model_root / "memory-summary.json").write_text(
        json.dumps(report_memory(memory), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (model_root / "mastery-summary.json").write_text(
        json.dumps(summarize_mastery(mastery, include_tasks=False), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (model_root / "QUALIFICATION_COMPLETE").touch()


def test_compare_requires_exact_paired_evidence_and_reports_direction(tmp_path: Path) -> None:
    _write_model(
        tmp_path,
        "base",
        [_memory_trace(idx=0, strict=1.0), _memory_trace(idx=1, strict=0.0)],
        [_mastery_trace(name="child-0", path_accesses=0.0)],
    )
    _write_model(
        tmp_path,
        "step-1",
        [_memory_trace(idx=0, strict=1.0), _memory_trace(idx=1, strict=1.0)],
        [_mastery_trace(name="child-0", path_accesses=1.0)],
    )

    result = compare(
        tmp_path,
        labels=("base", "step-1"),
        expected_memory_count=2,
        expected_mastery_count=1,
    )
    candidate = result["comparisons"]["step-1"]

    assert result["decision"] == "REVIEW_REQUIRED"
    assert result["typed_contract_violations"] == []
    assert candidate["memory_paired"]["overall"] == {
        "count": 2,
        "reference_success": 1,
        "candidate_success": 2,
        "gain": 1,
        "loss": 0,
        "both_success": 1,
        "both_failure": 0,
        "net_gain": 1,
    }
    assert candidate["memory_deltas"]["overall_means"]["strict_success"] == 0.5
    assert candidate["mastery_deltas"]["families"]["child"]["mean_deltas"]["parent_path_access"] == 1.0


def test_compare_fails_closed_on_incomplete_candidate(tmp_path: Path) -> None:
    (tmp_path / "base").mkdir()

    with pytest.raises(SystemExit, match="qualification is incomplete for base"):
        compare(tmp_path, labels=("base",), expected_memory_count=0, expected_mastery_count=0)


def test_compare_fails_closed_when_frozen_task_identities_differ(tmp_path: Path) -> None:
    _write_model(
        tmp_path,
        "base",
        [_memory_trace(idx=0, strict=1.0)],
        [_mastery_trace(name="child-0", path_accesses=0.0)],
    )
    _write_model(
        tmp_path,
        "step-1",
        [_memory_trace(idx=1, strict=1.0)],
        [_mastery_trace(name="child-0", path_accesses=0.0)],
    )

    with pytest.raises(SystemExit, match="frozen memory task identities differ"):
        compare(
            tmp_path,
            labels=("base", "step-1"),
            expected_memory_count=1,
            expected_mastery_count=1,
        )
