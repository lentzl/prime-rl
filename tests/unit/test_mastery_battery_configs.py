import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "configs" / "debug" / "subagent-communication"


def load_config(name: str) -> dict:
    with (CONFIG_ROOT / name).open("rb") as stream:
        return tomllib.load(stream)


def test_ownership_battery_allows_prime_agent_to_finish() -> None:
    names = (
        "274-qwen35-27b-mastery-ownership-child-ood.toml",
        "275-qwen35-27b-mastery-ownership-coordinator-ood.toml",
    )
    configs = [load_config(name) for name in names]
    limits = [config["env"]["agent"] for config in configs]

    assert all(limit["max_turns"] >= 8 for limit in limits)
    assert all(limit["max_output_tokens"] >= 24_576 for limit in limits)
    assert all(limit["max_total_tokens"] >= 65_536 for limit in limits)


def test_oolong_battery_leaves_room_for_the_agent_feedback_loop() -> None:
    config = load_config("276-qwen35-27b-mastery-oolong-ood.toml")
    taskset = config["env"]["taskset"]
    limits = config["env"]["agent"]

    assert taskset["context_len"] <= 16_384
    assert limits["max_turns"] >= 24
    assert limits["max_input_tokens"] >= 196_608
    assert limits["max_output_tokens"] >= 24_576
    assert limits["max_total_tokens"] >= 221_184


def test_teacher_collections_are_disjoint_and_keep_thinking_enabled() -> None:
    names = (
        "278-qwen35-27b-mastery-child-teacher-collection.toml",
        "279-qwen35-27b-mastery-coordinator-teacher-collection.toml",
    )
    configs = [load_config(name) for name in names]

    assert {config["env"]["taskset"]["ownership"] for config in configs} == {
        "child",
        "coordinator",
    }
    assert all(config["env"]["taskset"]["instance_offset"] >= 20_000 for config in configs)
    assert all(config["env"]["agent"]["harness"]["thinking"] == "high" for config in configs)
    assert all(config["sampling"]["temperature"] == 1.0 for config in configs)
    assert all(config["num_rollouts"] >= 16 for config in configs)


def test_guided_collection_is_complete_thinking_data_not_a_frozen_gate() -> None:
    config = load_config("280-qwen35-27b-mastery-guided-communication-collection.toml")
    taskset = config["env"]["taskset"]

    assert taskset["instruction_level"] == "guided"
    assert taskset["instance_offset"] >= 22_000
    assert set(taskset["families"]) == {
        "direct",
        "single",
        "parallel",
        "followup",
        "handshake",
    }
    assert config["env"]["agent"]["harness"]["thinking"] == "high"
    assert config["env"]["agent"]["harness"]["autonomous"] is True
    assert config["num_rollouts"] >= 4


def test_teacher_bootstraps_adapt_qwen35_linear_attention_and_preserve_thinking() -> None:
    names = (
        "281-qwen35-27b-prime-agent-teacher-bootstrap-r64.toml",
        "282-qwen35-27b-prime-agent-teacher-bootstrap-r128.toml",
    )
    configs = [load_config(name) for name in names]

    assert [config["model"]["lora"]["rank"] for config in configs] == [64, 128]
    assert all(config["deployment"]["num_gpus"] == 4 for config in configs)
    assert all(config["renderer"]["enable_thinking"] is True for config in configs)
    assert all(config["ckpt"]["interval"] <= 2 for config in configs)
    assert all(
        any("linear_attn" in target for target in config["model"]["lora"]["target_modules"]) for config in configs
    )
