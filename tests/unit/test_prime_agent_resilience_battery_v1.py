import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "experiments" / "qwen35-27b-prime-agent-resilience-v1"
CONFIG_NAMES = ("281-calibration.toml", "282-heldout.toml")


def _load(name: str) -> dict:
    with (EXPERIMENT / name).open("rb") as stream:
        return tomllib.load(stream)


def test_resilience_v1_is_a_disjoint_12_task_official_harness_battery() -> None:
    calibration, heldout = (_load(name) for name in CONFIG_NAMES)
    configs = (calibration, heldout)

    assert sum(config["num_tasks"] for config in configs) == 12
    assert calibration["env"]["taskset"]["split"] == "calibration"
    assert heldout["env"]["taskset"]["split"] == "heldout"
    assert calibration["env"]["taskset"]["instance_offset"] != (
        heldout["env"]["taskset"]["instance_offset"]
    )
    assert all(
        config["env"]["taskset"]["families"]
        == ["malformed_result_repair", "delayed_result", "message_type_repair"]
        for config in configs
    )
    assert all(
        config["env"]["agent"]["harness"]
        == {
            "id": "prime_agent",
            "version": "0.7.2-beta.495.1.97b994c",
        }
        for config in configs
    )
    assert all(config["sampling"]["reasoning_effort"] == "high" for config in configs)


def test_resilience_launcher_records_provenance_and_requires_valid_traces() -> None:
    launcher = (
        ROOT / "scripts" / "run_qwen35_27b_prime_agent_resilience_v1.sh"
    ).read_text()

    assert all(name.removesuffix(".toml") in launcher for name in CONFIG_NAMES)
    assert "git rev-parse HEAD" in launcher
    assert "git -C deps/verifiers rev-parse HEAD" in launcher
    assert "prime_agent_version=0.7.2-beta.495.1.97b994c" in launcher
    assert "--require-valid-traces" in launcher
    assert 'SUMMARY.json' in launcher


def test_model_launcher_can_target_the_separate_resilience_suite() -> None:
    launcher = (
        ROOT / "scripts" / "run_qwen35_27b_prime_agent_mastery_baseline_v2.sh"
    ).read_text()

    assert "EVAL_DRIVER" in launcher
    assert "EVAL_EXPERIMENT_DIR" in launcher
    assert '"$root"/"$eval_experiment"/*.toml' in launcher
