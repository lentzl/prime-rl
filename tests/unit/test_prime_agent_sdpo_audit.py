from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "experiments" / "qwen35-27b-prime-agent-sdpo-v1" / "zero-lr-audit.toml"


def test_zero_lr_audit_exercises_exact_typed_feedback_without_moving_weights() -> None:
    config = cli(RLConfig, args=["@", str(CONFIG), "--dry-run"])
    source = config.orchestrator.train.source[0]
    taskset = source.env.taskset
    algo = config.orchestrator.algo

    assert config.max_steps == 1
    assert config.max_train_batch_lead == 0
    assert config.trainer.optim.lr == 0.0
    assert config.trainer.model.fused_lm_head_token_chunk_size == "disabled"
    assert config.ckpt is not None and config.ckpt.interval is None
    assert config.model.name == "Qwen/Qwen3.5-27B"
    assert config.run.name == config.run.dir == "zero-lr-audit"
    assert config.clean is True
    assert algo.type == "sdpo"
    assert algo.require_explicit_feedback is True
    assert algo.required_feedback_contract_schema == ("prime-agent/ownership-decision-feedback/v1")
    assert algo.multi_turn_replay is True
    assert algo.filter is not None
    assert algo.filter.import_path == ("subagent_communication_v1.taskset.keep_first_coordinator_tool_call")
    assert taskset.id == "ownership-invariant-v1"
    assert taskset.ownership == "child"
    assert taskset.instruction_level == "standard"
    assert taskset.record_causal_feedback is True
    assert source.env.agent.harness.version == "0.7.2-beta.495.1.97b994c"
    assert config.orchestrator.train.sampling.reasoning_effort == "high"


def test_zero_lr_audit_launcher_fails_closed() -> None:
    launcher = (ROOT / "scripts" / "run_qwen35_27b_prime_agent_sdpo_zero_lr_audit_v1.sh").read_text()

    assert "refusing to launch while another GPU process is active" in launcher
    assert "zero-LR audit must run exactly one step" in launcher
    assert "zero-LR audit refuses a nonzero learning rate" in launcher
    assert "zero-LR audit must not write a checkpoint" in launcher
    assert "SDPO_AUDIT_DRY_RUN" in launcher
    assert "fc05daec18b0a78c049392ed2e771dde82bdf654" in launcher
    assert "snapshot_download" in launcher
    assert '"$(basename "$model_snapshot")" != "$model_revision"' in launcher
    assert 'rl @ "$config" --model.name "$model_snapshot"' in launcher
    assert "validate_prime_agent_sdpo_zero_lr_audit_v1.py" in launcher
    assert '--output "$run_dir/AUDIT.json"' in launcher
