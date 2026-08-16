from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli

ROOT = Path(__file__).parents[2]
CONFIG = (
    ROOT
    / "experiments"
    / "qwen35-27b-procedural-harness-master-v1"
    / "bootstrap-grpo.toml"
)
SHAPED_CONFIG = CONFIG.with_name("bootstrap-shaped-grpo.toml")
LAUNCHER = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_master_bootstrap_v1.sh"
SUMMARIZER = ROOT / "scripts" / "summarize_procedural_harness_master_v1.py"


def test_bootstrap_is_full_weight_hard_reward_grpo() -> None:
    config = cli(RLConfig, args=["@", str(CONFIG), "--dry-run"])
    source = config.orchestrator.train.source[0]

    assert config.max_steps == 4
    assert config.max_train_batch_lead == 0
    assert config.model.name == "Qwen/Qwen3.5-27B"
    assert config.trainer.model.lora is None
    assert config.trainer.model.optimization_dtype == "bfloat16"
    assert config.trainer.optim.type == "adamw"
    assert config.trainer.optim.lr == 5e-7
    assert config.trainer.ckpt is not None
    assert config.trainer.ckpt.weights_only is True
    assert config.trainer.ckpt.interval == 1
    assert config.orchestrator.batch_size == 16
    assert config.orchestrator.group_size == 8
    assert config.orchestrator.max_inflight_episodes == 8
    assert config.orchestrator.oversampling_factor == 0.5
    assert config.orchestrator.max_off_policy_steps == 0
    assert config.orchestrator.algo.type == "grpo"
    assert source.algo is not None and source.algo.type == "grpo"
    assert source.group_size == 8
    assert source.env.taskset.split == "train_gen"
    assert source.env.taskset.start_index == 200000
    assert set(source.env.taskset.families) == {
        "single",
        "parallel",
        "mixed",
        "followup",
        "verify",
    }
    assert source.env.agent.harness.version == "0.7.2-beta.495.1.97b994c"
    assert source.env.agent.harness.autonomous is True
    assert source.env.agent.harness.gates == [
        "python /workspace/.procedural-harness-master/completion_gate.py"
    ]
    assert config.orchestrator.renderer.enable_thinking is True
    assert config.inference is not None
    assert config.inference.vllm.max_model_len == 32768

    filters = {item.type: item for item in config.orchestrator.pre_batch_filters}
    assert filters["trainable_token_window"].enforce is True
    assert filters["trainable_token_window"].max_tokens == 8192
    assert filters["zero_advantage"].enforce is True


def test_bootstrap_uses_only_the_executable_environment_reward() -> None:
    taskset = (
        ROOT
        / "deps"
        / "verifiers"
        / "environments"
        / "procedural_harness_master_v1"
        / "procedural_harness_master_v1"
        / "taskset.py"
    ).read_text()
    config = CONFIG.read_text()

    assert '@vf.reward(weight=1.0)\n    async def harness_score' in taskset
    assert taskset.count("@vf.reward") == 1
    assert "harness_contract" in taskset
    assert "sdpo" not in config.lower()
    assert "lora" not in config.lower()


def test_constrained_bootstrap_is_isolated_from_hard_reward_run() -> None:
    hard = cli(RLConfig, args=["@", str(CONFIG), "--dry-run"])
    shaped = cli(RLConfig, args=["@", str(SHAPED_CONFIG), "--dry-run"])

    assert hard.orchestrator.train.source[0].env.taskset.task.reward_mode == "hard"
    assert shaped.orchestrator.train.source[0].env.taskset.task.reward_mode == "bootstrap"
    assert shaped.run.name == "bootstrap-shaped-grpo"
    assert shaped.trainer.model.lora is None
    assert shaped.trainer.optim.lr == hard.trainer.optim.lr


def test_admission_screen_is_disjoint_from_bootstrap_window() -> None:
    admission_start = 100000
    admission_count = 6
    admission_rollouts = 8
    config = cli(RLConfig, args=["@", str(CONFIG), "--dry-run"])
    taskset = config.orchestrator.train.source[0].env.taskset

    assert admission_start + admission_count <= taskset.start_index
    assert admission_rollouts == config.orchestrator.group_size


def test_bootstrap_launcher_fails_closed() -> None:
    launcher = LAUNCHER.read_text()
    selector = SUMMARIZER.read_text()

    assert "untouched-admission-r4" in launcher
    assert "untouched-admission-r3" not in launcher
    assert "untouched-admission-r2" not in launcher
    assert "select_training_mode" in launcher
    assert "admission must contain 48 error-free episodes" in selector
    assert "no non-direct informative hard or bootstrap comparison group" in selector
    assert "bootstrap-shaped-grpo.toml" in launcher
    assert 'mode=${selection%%|*}' in launcher
    assert "refusing to launch while another GPU process is active" in launcher
    assert "Qwen/Qwen3.5-27B" in launcher
    assert "fc05daec18b0a78c049392ed2e771dde82bdf654" in launcher
    assert "VLLM_USE_FLASHINFER_SAMPLER" in launcher
    assert 'rl @ "$config" --model.name "$model_snapshot"' in launcher
    assert "PROCEDURAL_HARNESS_TRAIN_DRY_RUN" in launcher
