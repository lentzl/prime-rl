import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "summarize_prime_agent_mastery.py"
SPEC = importlib.util.spec_from_file_location("summarize_prime_agent_mastery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trace(name: str, family: str, **metrics: float) -> dict:
    return {
        "ok": True,
        "task": {"data": {"name": name, "family": family}},
        "metrics": metrics,
        "rewards": {"task": {"score": metrics.get("answer_accuracy", 0.0)}},
        "stop_condition": "agent_completed",
    }


def test_summary_preserves_family_boundary_and_behavioral_issues() -> None:
    summary = MODULE.summarize(
        [
            trace(
                "direct-v0",
                "direct",
                answer_accuracy=1.0,
                protocol_aligned=1.0,
                clean_protocol_aligned=1.0,
            ),
            trace(
                "followup-v0",
                "followup",
                answer_accuracy=1.0,
                protocol_aligned=0.0,
                clean_protocol_aligned=0.0,
                natural_followup_causal=0.0,
                coordinator_delegated_path_accesses=1.0,
                roster_calls=2.0,
            ),
        ]
    )

    assert summary["trace_count"] == 2
    assert summary["families"]["direct"]["clean_count"] == 1
    assert summary["families"]["followup"]["clean_count"] == 0
    assert summary["tasks"][1]["issues"] == [
        "protocol",
        "clean",
        "causal",
        "path-access=1",
        "roster=2",
    ]


def test_reward_only_and_ownership_tasks_are_not_false_clean() -> None:
    oolong = trace("oolong-0", "oolong")
    ownership = trace(
        "child-heldout-0",
        "child",
        strict_success=0.0,
        state_retained=0.0,
        parent_path_access=1.0,
    )
    ownership["ok"] = False

    summary = MODULE.summarize([oolong, ownership])

    assert summary["tasks"][0]["issues"] == ["reward"]
    assert summary["tasks"][1]["issues"] == [
        "trace-error",
        "strict",
        "path-access=1",
    ]
    assert summary["families"]["oolong"]["clean_count"] == 0
    assert summary["families"]["child"]["clean_count"] == 0
