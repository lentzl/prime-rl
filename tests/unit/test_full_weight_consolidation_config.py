from pathlib import Path

from scripts.prepare_prime_agent_full_weight_consolidation import (
    BASE_REVISION,
    prepare,
)

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "configs/debug/subagent-communication/316-qwen35-27b-prime-agent-mastery-grpo.toml"


def test_consolidation_is_iterative_full_weight_and_removes_shortcut_pressure(tmp_path: Path) -> None:
    config = prepare(SOURCE, tmp_path / "resolved.toml", tmp_path / "training")

    assert config.max_steps == 8
    assert config.model.revision == BASE_REVISION
    assert config.deployment.num_train_gpus == 6
    assert config.deployment.num_infer_gpus == 2
    assert config.trainer.model.lora is None
    assert config.trainer.optim.lr == 5e-7
    assert config.orchestrator.algo.length_penalty is None
    assert config.orchestrator.eval is None
    assert config.ckpt.interval == 2


def test_consolidation_rebalances_fresh_executable_sources(tmp_path: Path) -> None:
    config = prepare(SOURCE, tmp_path / "resolved.toml", tmp_path / "training")
    sources = {source.name: source for source in config.orchestrator.train.source}

    assert set(sources) == {
        "mastery-foundations-train",
        "mastery-ownership-child-train",
        "mastery-ownership-coordinator-train",
        "mastery-routing-direct-single-train",
        "mastery-coupled-communication-train",
    }
    assert [sources[name].ratio for name in sources] == [1.0, 1.0, 1.0, 2.0, 3.0]
    assert sources["mastery-foundations-train"].env.taskset.families == (
        "ipython_cell",
        "persistence",
        "subagent_lifecycle",
        "harness_state",
    )
    assert sources["mastery-routing-direct-single-train"].env.taskset.families == ("direct", "single")
    assert sources["mastery-coupled-communication-train"].env.taskset.families == (
        "parallel",
        "followup",
        "handshake",
    )
    assert all("oolong" not in name for name in sources)
    assert all(source.env.taskset.instance_offset >= 1000 for source in sources.values())
