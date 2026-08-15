import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "summarize_prime_agent_mastery_v2.py"
SPEC = importlib.util.spec_from_file_location("summarize_prime_agent_mastery_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _trace(name: str, family: str, reward: float = 1.0, **metrics: float) -> dict:
    return {
        "ok": True,
        "task": {"data": {"name": name, "family": family}},
        "metrics": metrics,
        "rewards": {"task": {"score": reward}},
        "stop_condition": "agent_completed",
    }


def test_summary_preserves_foundation_and_coordination_failures() -> None:
    summary = MODULE.summarize(
        [
            _trace("ipython-cell-i0", "ipython_cell"),
            _trace(
                "followup-v0",
                "followup",
                answer_accuracy=1.0,
                protocol_aligned=0.0,
                clean_protocol_aligned=0.0,
                natural_followup_causal=0.0,
                bidirectional_control=0.0,
            ),
        ]
    )

    assert summary["trace_count"] == 2
    assert summary["issue_count"] == 1
    assert summary["families"]["ipython_cell"]["clean_count"] == 1
    assert summary["tasks"][1]["issues"] == [
        "protocol",
        "clean",
        "causal",
        "bidirectional-control",
    ]


def test_reward_only_foundation_and_oolong_failures_are_visible() -> None:
    summary = MODULE.summarize(
        [
            _trace("child-cancellation-i0", "child_cancellation", reward=0.0),
            _trace("oolong-0", "oolong", reward=0.25),
        ]
    )

    assert summary["issue_count"] == 2
    assert all(task["issues"] == ["reward"] for task in summary["tasks"])
    assert summary["families"]["oolong"]["mean_reward"] == 0.25


def test_load_traces_accepts_direct_and_enveloped_records(tmp_path: Path) -> None:
    direct = _trace("direct-v0", "direct")
    enveloped = _trace("parallel-v0", "parallel")
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(direct) + "\n" + json.dumps({"traces": [enveloped]}) + "\n")

    assert MODULE.load_traces([path]) == [direct, enveloped]


def test_oolong_family_is_recovered_from_the_task_type() -> None:
    trace = _trace("unknown", "unused")
    trace["task"] = {"type": "OolongSynthTask", "data": {"name": None}}

    summary = MODULE.summarize([trace])

    assert summary["tasks"][0]["family"] == "oolong"


def test_validity_gate_rejects_harness_errors() -> None:
    trace = _trace("child-v0", "child")
    trace.update(
        ok=False,
        id="trace-with-harness-error",
        errors=[{"type": "HarnessError", "message": "install failed"}],
    )

    with pytest.raises(SystemExit, match="trace-with-harness-error"):
        MODULE.require_valid_traces([trace])


def test_validity_gate_scores_model_rollout_budget_exhaustion() -> None:
    trace = _trace("oolong-200", "oolong", reward=0.0)
    trace.update(
        ok=False,
        stop_condition="error",
        errors=[
            {
                "type": "HarnessError",
                "message": "agent timeout: rollout exceeded its 1200s budget",
            }
        ],
    )

    MODULE.require_valid_traces([trace])
    summary = MODULE.summarize([trace])

    assert summary["issue_count"] == 1
    assert summary["tasks"][0]["issues"] == ["trace-error"]


@pytest.mark.parametrize(
    "message",
    [
        "Prime Agent artifact install failed",
        "container disappeared during rollout",
        "agent timeout: setup exceeded its 300s budget",
    ],
)
def test_validity_gate_still_rejects_non_behavioral_harness_failures(message: str) -> None:
    trace = _trace("invalid-runtime", "oolong", reward=0.0)
    trace.update(
        ok=False,
        id="invalid-runtime",
        stop_condition="error",
        errors=[{"type": "HarnessError", "message": message}],
    )

    with pytest.raises(SystemExit, match="invalid-runtime"):
        MODULE.require_valid_traces([trace])
