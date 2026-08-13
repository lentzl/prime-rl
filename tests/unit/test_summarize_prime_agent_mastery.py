import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "summarize_prime_agent_mastery.py"
SPEC = importlib.util.spec_from_file_location("summarize_prime_agent_mastery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trace(name: str, family: str, **metrics: float) -> dict:
    return {
        "ok": True,
        "info": {"env_name": "coordination", "policy_version": 8},
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
                bidirectional_control=0.0,
                coordinator_delegated_path_accesses=1.0,
                roster_calls=2.0,
            ),
        ]
    )

    assert summary["trace_count"] == 2
    assert summary["issue_count"] == 1
    assert summary["policy_versions"] == [8]
    assert summary["environments"]["coordination"] == {
        "count": 2,
        "clean_count": 1,
        "families": ["direct", "followup"],
    }
    assert summary["families"]["direct"]["clean_count"] == 1
    assert summary["families"]["followup"]["clean_count"] == 0
    assert summary["tasks"][1]["issues"] == [
        "protocol",
        "clean",
        "causal",
        "bidirectional-control",
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


def test_load_traces_accepts_direct_online_and_legacy_records(tmp_path: Path) -> None:
    direct = trace("direct-v0", "direct", answer_accuracy=1.0)
    legacy = trace("parallel-v0", "parallel", answer_accuracy=1.0)
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(direct) + "\n" + json.dumps({"traces": [legacy]}) + "\n",
        encoding="utf-8",
    )

    assert MODULE.load_traces([path]) == [direct, legacy]


def test_control_metrics_are_part_of_family_cleanliness() -> None:
    summary = MODULE.summarize(
        [
            trace(
                "single-v0",
                "single",
                answer_accuracy=1.0,
                protocol_aligned=1.0,
                clean_protocol_aligned=1.0,
                post_fan_in_control=0.5,
            ),
            trace(
                "handshake-v0",
                "handshake",
                answer_accuracy=1.0,
                protocol_aligned=1.0,
                clean_protocol_aligned=1.0,
                natural_followup_causal=1.0,
                bidirectional_control=0.75,
            ),
        ]
    )

    assert summary["tasks"][0]["issues"] == ["post-fan-in-control"]
    assert summary["tasks"][1]["issues"] == ["bidirectional-control"]


def test_unnamed_online_trace_uses_environment_as_family() -> None:
    online = trace("unused", "unused")
    online["task"]["data"] = {"name": None, "family": None}
    online["info"]["env_name"] = "oolong-externalization"
    online["ok"] = False

    summary = MODULE.summarize([online])

    assert summary["tasks"][0]["family"] == "oolong-externalization"
    assert summary["tasks"][0]["issues"] == ["trace-error"]


def test_no_visible_reply_is_classified_as_model_behavior() -> None:
    online = trace("oolong-0", "oolong")
    online["ok"] = False
    online["errors"] = [{"message": "ACP agent produced no visible reply (stop_reason=end_turn)"}]

    summary = MODULE.summarize([online])

    assert summary["tasks"][0]["issues"] == ["no-visible-reply"]


def test_coordinator_owned_path_access_is_not_reported_as_bypass() -> None:
    coordinator = trace(
        "coordinator-heldout-v0",
        "coordinator",
        strict_success=1.0,
        parent_path_access=1.0,
        direct_answer_accuracy=1.0,
    )

    summary = MODULE.summarize([coordinator])

    assert summary["tasks"][0]["issues"] == []


def test_policy_version_summaries_do_not_blend_checkpoints() -> None:
    base = trace("direct-base", "direct", answer_accuracy=0.0)
    base["info"]["policy_version"] = 0
    checkpoint = trace("direct-step-8", "direct", answer_accuracy=1.0)

    summaries = MODULE.summarize_by_policy_version([checkpoint, base])

    assert list(summaries) == ["0", "8"]
    assert summaries["0"]["trace_count"] == 1
    assert summaries["0"]["families"]["direct"]["means"]["answer_accuracy"] == 0.0
    assert summaries["8"]["families"]["direct"]["means"]["answer_accuracy"] == 1.0


def test_summary_only_retains_issue_count_without_task_records() -> None:
    summary = MODULE.summarize(
        [trace("direct-v0", "direct", answer_accuracy=0.0)],
        include_tasks=False,
    )

    assert summary["trace_count"] == 1
    assert summary["issue_count"] == 1
    assert "tasks" not in summary
