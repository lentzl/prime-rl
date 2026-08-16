import json
from pathlib import Path

import pytest

import scripts.summarize_procedural_harness_master_v1 as summary_module
from scripts.summarize_procedural_harness_master_v1 import (
    configured_reward_mode,
    select_training_mode,
    summarize,
)

ROOT = Path(__file__).parents[2]


def _trace(
    family: str,
    score: float,
    episode_id: str | None = None,
    bootstrap_progress: float | None = None,
) -> dict:
    return {
        "task": {"data": {"family": family, "episode_id": episode_id or family}},
        "rewards": {"harness_score": {"score": score, "weight": 1.0}},
        "metrics": {
            "final_answer_exact": score,
            "all_required_atoms": score,
            "no_forbidden_atoms": 1.0,
            "ordering_satisfied": score,
            "cardinality_exact": score,
            "required_atoms_fraction": score,
            "bootstrap_progress": score if bootstrap_progress is None else bootstrap_progress,
        },
        "errors": [],
    }


def test_summarize_flattens_episode_envelopes(tmp_path) -> None:
    run = tmp_path / "valid"
    run.mkdir()
    episodes = [
        {"id": "a", "traces": [_trace("direct", 1.0)]},
        {"id": "b", "traces": [_trace("parallel", 0.0)]},
    ]
    (run / "traces.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes)
    )

    report = summarize([run])

    assert report["rescored"] is False
    assert report["episodes"] == 2
    assert report["harness"] == {"episodes": 2, "passed": 1, "rate": 0.5}
    assert report["by_family"]["direct"]["rate"] == 1.0
    assert report["by_family"]["parallel"]["rate"] == 0.0
    assert report["comparison_groups"] == {
        "groups": 2,
        "informative": 0,
        "all_pass": 1,
        "all_fail": 1,
    }
    assert report["bootstrap_comparison_groups"] == {
        "groups": 2,
        "informative": 0,
        "homogeneous": 2,
        "all_zero": 1,
    }


def test_summarize_identifies_informative_grpo_groups(tmp_path) -> None:
    run = tmp_path / "admission"
    run.mkdir()
    episodes = [
        {"id": "a0", "traces": [_trace("single", 0.0, "single-1")]},
        {"id": "a1", "traces": [_trace("single", 1.0, "single-1")]},
        {"id": "b0", "traces": [_trace("direct", 1.0, "direct-1")]},
        {"id": "b1", "traces": [_trace("direct", 1.0, "direct-1")]},
    ]
    (run / "traces.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes)
    )

    report = summarize([run])

    assert report["rescored"] is False
    assert report["comparison_groups"] == {
        "groups": 2,
        "informative": 1,
        "all_pass": 1,
        "all_fail": 0,
    }
    assert report["by_family_groups"]["single"]["informative"] == 1
    assert report["by_family_groups"]["direct"]["all_pass"] == 1


def test_summarize_identifies_bootstrap_signal_in_hard_failure_group(tmp_path) -> None:
    run = tmp_path / "admission"
    run.mkdir()
    episodes = [
        {"id": "a0", "traces": [_trace("single", 0.0, "single-1", 0.25)]},
        {"id": "a1", "traces": [_trace("single", 0.0, "single-1", 0.50)]},
    ]
    (run / "traces.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes)
    )

    report = summarize([run])

    assert report["by_family_groups"]["single"]["informative"] == 0
    assert report["by_family_bootstrap_groups"]["single"] == {
        "groups": 1,
        "informative": 1,
        "homogeneous": 0,
        "all_zero": 0,
    }


def test_summarize_uses_rescored_hard_gate_when_requested(tmp_path, monkeypatch) -> None:
    run = tmp_path / "rescored"
    run.mkdir()
    (run / "traces.jsonl").write_text(
        json.dumps({"id": "a", "traces": [_trace("single", 0.0)]}) + "\n"
    )

    def pass_trace(trace):
        trace["rewards"]["harness_score"]["score"] = 1.0
        return trace

    monkeypatch.setattr(summary_module, "_rescore", pass_trace)

    report = summarize([run], rescore=True)

    assert report["rescored"] is True
    assert report["harness"]["passed"] == 1


def _admission_report() -> dict:
    families = {"direct", "single", "parallel", "mixed", "followup", "verify"}
    return {
        "rescored": True,
        "episodes": 48,
        "errors": 0,
        "by_family": {family: {"episodes": 8} for family in families},
        "by_family_groups": {
            family: {"groups": 1, "informative": 0} for family in families
        },
        "by_family_bootstrap_groups": {
            family: {"groups": 1, "informative": 0} for family in families
        },
    }


def test_training_mode_prefers_hard_reward_signal() -> None:
    report = _admission_report()
    report["by_family_groups"]["single"]["informative"] = 1
    report["by_family_bootstrap_groups"]["parallel"]["informative"] = 1

    assert select_training_mode(report) == ("hard", ["single"])


def test_training_mode_falls_back_to_measured_bootstrap_signal() -> None:
    report = _admission_report()
    report["by_family_bootstrap_groups"]["verify"]["informative"] = 1

    assert select_training_mode(report) == ("bootstrap", ["verify"])


def test_training_mode_rejects_incomplete_admission() -> None:
    report = _admission_report()
    report["episodes"] = 47

    with pytest.raises(ValueError, match="48 error-free episodes"):
        select_training_mode(report)


def test_training_config_reward_mode_honors_hard_default_and_shaped_override() -> None:
    experiment = ROOT / "experiments" / "qwen35-27b-procedural-harness-master-v1"

    assert configured_reward_mode(experiment / "bootstrap-grpo.toml") == "hard"
    assert configured_reward_mode(experiment / "bootstrap-shaped-grpo.toml") == "bootstrap"
