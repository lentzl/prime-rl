import random
from collections import Counter
from pathlib import Path

from subagent_communication_v1.taskset import SubagentCommunicationTaskset

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "experiments" / "qwen35-27b-prime-agent-sdpo-v1" / "zero-lr-audit.toml"


def test_zero_lr_audit_mixes_exact_typed_sdpo_with_grpo_retention() -> None:
    config = cli(RLConfig, args=["@", str(CONFIG), "--dry-run"])
    sources = {source.name: source for source in config.orchestrator.train.source}
    diagnostic = sources["ownership-child-diagnostic-sdpo"]
    taskset = diagnostic.env.taskset
    algo = diagnostic.algo

    assert config.max_steps == 1
    assert config.seq_len == 8192
    assert config.trainer.model.seq_len == 8192
    assert config.max_train_batch_lead == 0
    assert config.trainer.optim.lr == 0.0
    assert config.trainer.enable_token_export is True
    assert config.trainer.model.fused_lm_head_token_chunk_size == "disabled"
    assert config.trainer.model.fsdp_cpu_offload is True
    assert config.trainer.model.optim_cpu_offload is False
    assert config.ckpt is None
    assert config.model.name == "Qwen/Qwen3.5-27B"
    assert config.run.name == config.run.dir == "zero-lr-audit"
    assert config.clean is True
    assert config.orchestrator.batch_size == 16
    assert config.orchestrator.max_inflight_episodes == 8
    assert config.orchestrator.oversampling_factor == 0.5
    assert config.orchestrator.algo.type == "grpo"
    assert config.orchestrator.train.sampling.max_completion_tokens == 1024
    token_window = next(
        filt
        for filt in config.orchestrator.pre_batch_filters
        if filt.type == "trainable_token_window"
    )
    assert token_window.enforce is True
    assert token_window.max_tokens == config.trainer.model.seq_len
    assert set(sources) == {
        "ownership-child-diagnostic-sdpo",
        "ownership-coordinator-retention",
        "communication-direct-retention",
        "communication-single-retention",
        "communication-parallel-retention",
        "communication-causal-retention",
    }
    assert algo is not None and algo.type == "sdpo"
    assert algo.require_explicit_feedback is True
    assert algo.required_feedback_contract_schema == ("prime-agent/ownership-decision-feedback/v1")
    assert algo.multi_turn_replay is True
    assert algo.filter is not None
    assert algo.filter.import_path == ("subagent_communication_v1.taskset.keep_first_coordinator_tool_call")
    assert taskset.id == "ownership-invariant-v1"
    assert taskset.ownership == "child"
    assert taskset.instruction_level == "standard"
    assert taskset.record_causal_feedback is True
    assert diagnostic.group_size == 1
    for name, source in sources.items():
        assert source.env.agent.harness.version == "0.7.2-beta.495.1.97b994c"
        if name != diagnostic.name:
            assert source.algo is not None and source.algo.type == "grpo"
            assert source.group_size == 2
    assert sources["communication-causal-retention"].env.taskset.families == (
        "followup",
        "handshake",
    )
    assert {name: source.ratio for name, source in sources.items()} == {
        "ownership-child-diagnostic-sdpo": 4.0,
        "ownership-coordinator-retention": 2.0,
        "communication-direct-retention": 2.0,
        "communication-single-retention": 2.0,
        "communication-parallel-retention": 1.0,
        "communication-causal-retention": 4.0,
    }
    assert config.orchestrator.train.sampling.reasoning_effort == "high"


def test_zero_lr_audit_fixed_seed_allocates_both_causal_families() -> None:
    config = cli(RLConfig, args=["@", str(CONFIG), "--dry-run"])
    configured_sources = config.orchestrator.train.source
    rng = random.Random(42)
    allocation: Counter[str] = Counter()
    remaining = config.orchestrator.batch_size
    while remaining > 0:
        source = rng.choices(
            configured_sources,
            weights=[candidate.ratio for candidate in configured_sources],
            k=1,
        )[0]
        assert source.group_size <= remaining
        allocation[source.name] += source.group_size
        remaining -= source.group_size

    assert allocation == {
        "ownership-child-diagnostic-sdpo": 4,
        "ownership-coordinator-retention": 2,
        "communication-direct-retention": 2,
        "communication-single-retention": 2,
        "communication-parallel-retention": 2,
        "communication-causal-retention": 4,
    }
    causal = next(
        source
        for source in configured_sources
        if source.name == "communication-causal-retention"
    )
    tasks = SubagentCommunicationTaskset(causal.env.taskset).load()
    random.Random(1).shuffle(tasks)
    assert {task.data.family for task in tasks[:2]} == {"followup", "handshake"}


def test_zero_lr_audit_launcher_fails_closed() -> None:
    launcher = (ROOT / "scripts" / "run_qwen35_27b_prime_agent_sdpo_zero_lr_audit_v1.sh").read_text()

    assert "refusing to launch while another GPU process is active" in launcher
    assert "zero-LR audit must run exactly one step" in launcher
    assert "zero-LR audit refuses a nonzero learning rate" in launcher
    assert "zero-LR audit must omit checkpoint configuration" in launcher
    assert "zero-LR audit must use the 16-trace mechanism batch" in launcher
    assert "qualified 8192-token L40S pack ceiling" in launcher
    assert "zero-LR audit must cap concurrency at eight episodes" in launcher
    assert "SDPO_AUDIT_DRY_RUN" in launcher
    assert "fc05daec18b0a78c049392ed2e771dde82bdf654" in launcher
    assert "snapshot_download" in launcher
    assert '"$(basename "$model_snapshot")" != "$model_revision"' in launcher
    assert 'rl @ "$config" --model.name "$model_snapshot"' in launcher
    assert "validate_prime_agent_sdpo_zero_lr_audit_v1.py" in launcher
    assert '--output "$run_dir/AUDIT.json"' in launcher
