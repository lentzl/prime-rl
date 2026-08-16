import json

from scripts.summarize_procedural_harness_master_v1 import summarize


def _trace(family: str, score: float, episode_id: str | None = None) -> dict:
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

    assert report["comparison_groups"] == {
        "groups": 2,
        "informative": 1,
        "all_pass": 1,
        "all_fail": 0,
    }
    assert report["by_family_groups"]["single"]["informative"] == 1
    assert report["by_family_groups"]["direct"]["all_pass"] == 1
