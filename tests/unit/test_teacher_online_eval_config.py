from pathlib import Path

from prime_rl.configs.sft import SFTConfig
from prime_rl.entrypoints.sft import build_evaluator_config
from prime_rl.utils.config import cli

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/debug/subagent-communication/283-qwen35-27b-prime-agent-teacher-bootstrap-online.toml"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"


def test_teacher_bootstrap_online_eval_is_fully_resolved() -> None:
    config = cli(SFTConfig, args=["@", str(CONFIG), "--dry-run"])

    assert config.deployment.num_gpus == 2
    assert config.deployment.num_infer_gpus == 2
    assert config.model.revision == REVISION
    assert config.tokenizer.revision == REVISION
    assert config.inference is not None
    assert config.inference.vllm.revision == REVISION
    assert config.eval is not None
    assert config.eval.client.base_url == "http://localhost:8100/v1"
    assert config.eval.client.admin_base_url == ["http://localhost:8100/v1"]
    assert len(config.eval.source) == 5

    evaluator = build_evaluator_config(config)
    assert evaluator.model.name == config.model.name
    assert evaluator.model.revision == REVISION
    assert evaluator.model.trust_remote_code == config.model.trust_remote_code
