import json
from pathlib import Path

import pytest

from scripts.compare_natural_yield_sdpo_gates_v1 import GATES, compare
from scripts.compare_natural_yield_sdpo_target_gates_v1 import compare_target
from scripts.compare_prime_agent_runtime_natural_yield_v1 import compare_runtimes


def _write_gate(
    root: Path,
    label: str,
    suffix: str,
    family: str,
    passed: int,
    *,
    exact: float = 1.0,
    local_work_before_yield: float = 1.0,
    premature_yield_before_local_work: float = 0.0,
    forbidden_post_spawn_tool_before_child: float = 1.0,
    task_index: int = 1,
    max_concurrent: int = 1,
    client_base_url: str = "http://127.0.0.1:8100/v1",
    runtime_version: str = "0.7.2-test.old",
) -> None:
    gate = root / f"{label}-{suffix}" / "train-admission"
    gate.mkdir(parents=True)
    summary = {
        "rescored": True,
        "episodes": 8,
        "errors": 0,
        "harness": {"episodes": 8, "passed": passed, "rate": passed / 8},
        "diagnostic_means": {
            "final_answer_exact": exact,
            "local_work_before_yield": local_work_before_yield,
            "premature_yield_before_local_work": premature_yield_before_local_work,
            "forbidden_post_spawn_tool_before_child": (
                forbidden_post_spawn_tool_before_child
            ),
        },
    }
    (gate / "SUMMARY.json").write_text(json.dumps(summary))
    task = {
        "episode_id": f"train_gen-{family}-{task_index:08d}",
        "family": family,
        "prompt": f"perform {family}",
        "system_prompt": "use the harness",
        "oracle": {"expected_route": family},
    }
    traces = [{"task": {"data": task}} for _ in range(8)]
    (gate / "traces.jsonl").write_text(json.dumps({"traces": traces}) + "\n")
    configs = gate / "configs"
    configs.mkdir()
    (configs / "eval.json").write_text(
        json.dumps(
            {
                "model": f"/models/{label}",
                "output_dir": str(root / label),
                "max_concurrent": max_concurrent,
                "client": {"base_url": client_base_url, "type": "eval"},
                "env": {
                    "agent": {
                        "harness": {
                            "id": "prime_agent",
                            "version": runtime_version,
                        },
                        "runtime": {
                            "type": "docker",
                            "image": (
                                f"rlm-prime-agent-runtime:{runtime_version}-node22.19.0"
                            ),
                        },
                    }
                },
                "sampling": {"temperature": 1.0},
            }
        )
    )


def _write_battery(
    root: Path,
    label: str,
    scores: dict[str, int],
    *,
    exact: float = 1.0,
    changed_gate: str | None = None,
    runtime_version: str = "0.7.2-test.old",
) -> None:
    families = {
        "natural_yield": "natural_n1",
        "natural_yield_local_work": "natural_n1",
        "atomic_state": "atomic_state",
        "atomic_send": "atomic_send",
    }
    for name, suffix in GATES.items():
        _write_gate(
            root,
            label,
            suffix,
            families[name],
            scores[name],
            exact=exact if name == "natural_yield" else 1.0,
            task_index=2 if name == changed_gate else 1,
            runtime_version=runtime_version,
        )


def test_comparison_authorizes_replication_only_after_target_gain_and_retention(
    tmp_path: Path,
) -> None:
    _write_battery(
        tmp_path,
        "r7",
        {
            "natural_yield": 0,
            "natural_yield_local_work": 2,
            "atomic_state": 8,
            "atomic_send": 5,
        },
    )
    _write_battery(
        tmp_path,
        "candidate",
        {
            "natural_yield": 2,
            "natural_yield_local_work": 2,
            "atomic_state": 8,
            "atomic_send": 6,
        },
    )

    report = compare(tmp_path, "r7", "candidate")

    assert report["decision"]["eligible_for_independent_replication"] is True
    assert report["decision"]["promoted"] is False
    assert report["gates"]["natural_yield"]["delta_passed"] == 2


def test_comparison_rejects_prerequisite_or_exact_answer_regression(tmp_path: Path) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(
        tmp_path,
        "candidate",
        {
            "natural_yield": 2,
            "natural_yield_local_work": 2,
            "atomic_state": 8,
            "atomic_send": 4,
        },
        exact=0.75,
    )

    report = compare(tmp_path, "r7", "candidate")

    assert report["decision"]["target_improved"] is True
    assert report["decision"]["prerequisites_retained"] is False
    assert report["decision"]["target_exact_answer_not_regressed"] is False
    assert report["decision"]["eligible_for_independent_replication"] is False


def test_comparison_accepts_material_target_transition_reduction(tmp_path: Path) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(tmp_path, "candidate", scores)
    gate = (
        tmp_path
        / "candidate-natural-yield"
        / "train-admission"
        / "SUMMARY.json"
    )
    summary = json.loads(gate.read_text())
    summary["diagnostic_means"]["forbidden_post_spawn_tool_before_child"] = 0.75
    gate.write_text(json.dumps(summary))

    report = compare(tmp_path, "r7", "candidate")

    assert report["decision"]["target_hard_improved"] is False
    assert report["decision"]["target_forbidden_transition_reduced_materially"] is True
    assert report["decision"]["eligible_for_independent_replication"] is True


def test_comparison_fails_closed_on_different_task_draw(tmp_path: Path) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(tmp_path, "candidate", scores, changed_gate="natural_yield")

    with pytest.raises(ValueError, match="task specifications differ"):
        compare(tmp_path, "r7", "candidate")


def test_comparison_fails_closed_on_different_runtime_config(tmp_path: Path) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(tmp_path, "candidate", scores)
    config = (
        tmp_path
        / "candidate-natural-yield"
        / "train-admission"
        / "configs"
        / "eval.json"
    )
    resolved = json.loads(config.read_text())
    resolved["max_concurrent"] = 2
    config.write_text(json.dumps(resolved))

    with pytest.raises(ValueError, match="resolved evaluation configs differ"):
        compare(tmp_path, "r7", "candidate")


def test_comparison_rejects_premature_yield_regression(tmp_path: Path) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(tmp_path, "candidate", scores)
    gate = (
        tmp_path
        / "candidate-natural-yield-local-work"
        / "train-admission"
        / "SUMMARY.json"
    )
    summary = json.loads(gate.read_text())
    summary["diagnostic_means"]["premature_yield_before_local_work"] = 0.25
    gate.write_text(json.dumps(summary))

    report = compare(tmp_path, "r7", "candidate")

    assert report["decision"]["anti_overgeneralization_retained"] is False
    assert report["decision"]["eligible_for_independent_replication"] is False


@pytest.mark.parametrize(
    "diagnostic",
    ["local_work_before_yield", "premature_yield_before_local_work"],
)
def test_comparison_fails_closed_when_local_work_diagnostic_is_missing(
    tmp_path: Path,
    diagnostic: str,
) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(tmp_path, "candidate", scores)
    gate = (
        tmp_path
        / "candidate-natural-yield-local-work"
        / "train-admission"
        / "SUMMARY.json"
    )
    summary = json.loads(gate.read_text())
    del summary["diagnostic_means"][diagnostic]
    gate.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match=f"missing diagnostics: {diagnostic}"):
        compare(tmp_path, "r7", "candidate")


def test_comparison_fails_closed_when_target_diagnostic_is_missing(
    tmp_path: Path,
) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "r7", scores)
    _write_battery(tmp_path, "candidate", scores)
    gate = tmp_path / "candidate-natural-yield" / "train-admission" / "SUMMARY.json"
    summary = json.loads(gate.read_text())
    del summary["diagnostic_means"]["forbidden_post_spawn_tool_before_child"]
    gate.write_text(json.dumps(summary))

    with pytest.raises(
        ValueError,
        match="missing diagnostics: forbidden_post_spawn_tool_before_child",
    ):
        compare(tmp_path, "r7", "candidate")


def test_target_comparison_only_authorizes_retention_after_material_gain(
    tmp_path: Path,
) -> None:
    _write_gate(tmp_path, "r7", "natural-yield", "natural_n1", 0)
    _write_gate(
        tmp_path,
        "step2",
        "natural-yield",
        "natural_n1",
        0,
        forbidden_post_spawn_tool_before_child=0.75,
        client_base_url="http://127.0.0.1:8200/v1",
    )

    report = compare_target(tmp_path, "r7", "step2")

    assert report["decision"]["target_hard_improved"] is False
    assert report["decision"][
        "target_forbidden_transition_reduced_materially"
    ] is True
    assert report["decision"]["eligible_for_retention_gates"] is True
    assert report["decision"]["promoted"] is False


def test_target_comparison_rejects_exact_answer_regression(tmp_path: Path) -> None:
    _write_gate(tmp_path, "r7", "natural-yield", "natural_n1", 0)
    _write_gate(
        tmp_path,
        "step4",
        "natural-yield",
        "natural_n1",
        1,
        exact=0.875,
    )

    report = compare_target(tmp_path, "r7", "step4")

    assert report["decision"]["target_improved"] is True
    assert report["decision"]["target_exact_answer_not_regressed"] is False
    assert report["decision"]["eligible_for_retention_gates"] is False


def test_runtime_comparison_authorizes_replication_after_causal_gain(
    tmp_path: Path,
) -> None:
    base_scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "old", base_scores, runtime_version="0.7.2-test.old")
    _write_battery(
        tmp_path,
        "current",
        {
            **base_scores,
            "natural_yield": 2,
            "atomic_send": 6,
        },
        runtime_version="0.7.3-test.current",
    )

    report = compare_runtimes(
        tmp_path,
        "old",
        "current",
        expected_base_version="0.7.2-test.old",
        expected_candidate_version="0.7.3-test.current",
    )

    assert report["decision"]["current_runtime_connects_natural_yield"] is True
    assert report["decision"]["prerequisites_retained"] is True
    assert report["decision"]["spawn_then_local_work_retained"] is True
    assert report["decision"]["eligible_for_current_runtime_replication"] is True
    assert report["decision"]["weights_changed"] is False


def test_runtime_comparison_rejects_non_runtime_config_difference(
    tmp_path: Path,
) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "old", scores, runtime_version="0.7.2-test.old")
    _write_battery(
        tmp_path,
        "current",
        scores,
        runtime_version="0.7.3-test.current",
    )
    config = (
        tmp_path
        / "current-natural-yield"
        / "train-admission"
        / "configs"
        / "eval.json"
    )
    resolved = json.loads(config.read_text())
    resolved["max_concurrent"] = 2
    config.write_text(json.dumps(resolved))

    with pytest.raises(ValueError, match="differ beyond Prime Agent runtime identity"):
        compare_runtimes(tmp_path, "old", "current")


def test_runtime_comparison_requires_distinct_versions(tmp_path: Path) -> None:
    scores = {
        "natural_yield": 0,
        "natural_yield_local_work": 2,
        "atomic_state": 8,
        "atomic_send": 5,
    }
    _write_battery(tmp_path, "old", scores)
    _write_battery(tmp_path, "current", scores)

    with pytest.raises(ValueError, match="same Prime Agent version"):
        compare_runtimes(tmp_path, "old", "current")
