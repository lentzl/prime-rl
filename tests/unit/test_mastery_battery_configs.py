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
