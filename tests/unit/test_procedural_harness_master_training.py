from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli
from scripts.summarize_procedural_harness_master_v1 import (
    classify_curriculum_rung_admission,
    select_curriculum_rung_admission,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "experiments" / "qwen35-27b-procedural-harness-master-v1" / "bootstrap-grpo.toml"
SHAPED_CONFIG = CONFIG.with_name("bootstrap-shaped-grpo.toml")
ACTION_CONFIG = CONFIG.with_name("harness-action-grpo.toml")
ACTION_ADMISSION_CONFIG = CONFIG.with_name("harness-action-admission.toml")
LAUNCHER = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_master_bootstrap_v1.sh"
ACTION_ADMISSION_LAUNCHER = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_action_admission_v1.sh"
ACTION_TRAIN_LAUNCHER = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_action_grpo_v1.sh"
SUMMARIZER = ROOT / "scripts" / "summarize_procedural_harness_master_v1.py"
BASELINE = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_master_baseline_v1.sh"
CHECKPOINT_BATTERY = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_master_checkpoint_battery_v1.sh"


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
    assert source.env.agent.harness.gates == ["python /workspace/.procedural-harness-master/completion_gate.py"]
    assert config.orchestrator.renderer.enable_thinking is True
    assert config.inference is not None
    assert config.inference.vllm.max_model_len == 32768
    assert config.inference.vllm.gpu_memory_utilization == 0.80

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

    assert "@vf.reward(weight=1.0)\n    async def harness_score" in taskset
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
    assert shaped.orchestrator.group_size == 8
    assert shaped.orchestrator.oversampling_factor == 0.5


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

    assert "untouched-admission-v2-r1" in launcher
    assert "untouched-admission-r5" not in launcher
    assert "untouched-admission-r4" not in launcher
    assert "untouched-admission-r3" not in launcher
    assert "untouched-admission-r2" not in launcher
    assert "select_training_mode" in launcher
    assert "admission must contain 48 error-free episodes" in selector
    assert "no non-direct informative hard or bootstrap comparison group" in selector
    assert "bootstrap-shaped-grpo.toml" in launcher
    assert "mode=${selection%%|*}" in launcher
    assert "config=$resolved_config" in launcher
    assert 'families = sys.argv[3].split(",")' in launcher
    assert 'r"^families = \\[.*\\]$"' in launcher
    assert "refusing to launch while another GPU process is active" in launcher
    assert "Qwen/Qwen3.5-27B" in launcher
    assert "fc05daec18b0a78c049392ed2e771dde82bdf654" in launcher
    assert "VLLM_USE_FLASHINFER_SAMPLER" in launcher
    assert "uv sync --frozen --inexact --extra flash-attn" in launcher
    assert "vllm_router-0.2.0-cp38-abi3-manylinux_2_28_x86_64.whl" in launcher
    assert 'python -c "import prime_rl.trainer.model"' in launcher
    assert 'rl @ "$config" --model.name "$model_snapshot"' in launcher
    assert "PROCEDURAL_HARNESS_TRAIN_DRY_RUN" in launcher


def test_checkpoint_battery_evaluates_untouched_and_every_stable_step() -> None:
    launcher = CHECKPOINT_BATTERY.read_text()
    baseline = BASELINE.read_text()

    assert 'tomllib.load(handle)["max_steps"]' in launcher
    assert 'models+=("$model_snapshot")' in launcher
    assert 'labels+=("untouched")' in launcher
    assert 'for step in $(seq 1 "$max_steps")' in launcher
    assert 'if [[ ! -f "$weights/STABLE" ]]' in launcher
    assert "model.safetensors.index.json" in launcher
    assert "refusing to evaluate while another GPU process is active" in launcher
    assert "compare_procedural_harness_master_checkpoints_v1.py" in launcher
    assert "eval_experiment=experiments/qwen35-27b-procedural-harness-master-v1" in launcher
    assert 'EVAL_EXPERIMENT_DIR="$eval_experiment"' in launcher
    assert 'EVAL_EXPERIMENT_DIR="$experiment"' not in launcher
    assert "PRIME_MASTERY_OUTPUT_ROOT" in baseline


def test_curriculum_admission_requires_one_informative_hard_group() -> None:
    report = {
        "rescored": True,
        "episodes": 8,
        "errors": 0,
        "by_family": {"atomic_send": {"episodes": 8, "passed": 3, "rate": 0.375}},
        "by_family_groups": {
            "atomic_send": {
                "groups": 1,
                "informative": 1,
                "all_pass": 0,
                "all_fail": 0,
            }
        },
    }

    assert select_curriculum_rung_admission(report, "atomic_send") == "atomic_send"

    report["by_family_groups"]["atomic_send"]["informative"] = 0
    report["by_family_groups"]["atomic_send"]["all_fail"] = 1
    assert classify_curriculum_rung_admission(report, "atomic_send") == "disconnected"

    report["by_family_groups"]["atomic_send"]["all_fail"] = 0
    report["by_family_groups"]["atomic_send"]["all_pass"] = 1
    assert classify_curriculum_rung_admission(report, "atomic_send") == "mastered"


def test_harness_action_ramp_is_full_weight_hard_grpo() -> None:
    config = cli(RLConfig, args=["@", str(ACTION_CONFIG), "--dry-run"])
    source = config.orchestrator.train.source[0]

    assert config.max_steps == 4
    assert config.max_train_batch_lead == 0
    assert config.trainer.model.lora is None
    assert config.trainer.model.optimization_dtype == "bfloat16"
    assert config.trainer.optim.type == "adamw"
    assert config.trainer.optim.lr == 5e-7
    assert config.orchestrator.algo.type == "grpo"
    assert config.orchestrator.batch_size == 16
    assert config.orchestrator.group_size == 8
    assert config.orchestrator.max_inflight_episodes == 8
    assert source.group_size == 8
    assert source.env.taskset.curriculum_rung == "atomic_state"
    assert source.env.taskset.task.reward_mode == "hard"
    assert source.env.agent.max_turns == 16
    assert source.env.agent.harness.autonomous_max_turns == 16
    assert source.env.agent.timeout.rollout == 900.0
    filters = {item.type: item for item in config.orchestrator.pre_batch_filters}
    assert filters["zero_advantage"].enforce is True


def test_harness_action_launchers_are_variance_gated_and_cumulative() -> None:
    admission_config = ACTION_ADMISSION_CONFIG.read_text()
    admission_launcher = ACTION_ADMISSION_LAUNCHER.read_text()
    train_launcher = ACTION_TRAIN_LAUNCHER.read_text()
    selector = SUMMARIZER.read_text()

    assert "num_tasks = 1" in admission_config
    assert "num_rollouts = 8" in admission_config
    assert 'curriculum_rung = "atomic_state"' in admission_config
    assert "classify_curriculum_rung_admission" in admission_launcher
    assert "select_curriculum_rung_admission" in train_launcher
    assert "curriculum admission must contain eight error-free episodes" in selector
    assert 'status != "trainable"' in selector
    assert "HARNESS_ACTION_MODEL_PATH" in train_launcher
    assert "HARNESS_ACTION_MODEL_REPO" in train_launcher
    assert "refusing to launch while another GPU process is active" in train_launcher
    assert "bootstrap-shaped-grpo" not in train_launcher
    assert "HARNESS_ACTION_TRAIN_DRY_RUN" in train_launcher
    assert 'rl @ "$config" --model.name "$model_snapshot"' in train_launcher
