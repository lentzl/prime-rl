from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli
from scripts.audit_procedural_harness_event_control_support_v1 import (
    summarize_support,
    validate_support,
)
from scripts.summarize_procedural_harness_master_v1 import (
    classify_curriculum_rung_admission,
    select_curriculum_rung_admission,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "experiments" / "qwen35-27b-procedural-harness-master-v1" / "bootstrap-grpo.toml"
SHAPED_CONFIG = CONFIG.with_name("bootstrap-shaped-grpo.toml")
ACTION_CONFIG = CONFIG.with_name("harness-action-grpo.toml")
CUMULATIVE_ACTION_CONFIG = CONFIG.with_name("harness-send-followup-cumulative-grpo.toml")
ACTION_ADMISSION_CONFIG = CONFIG.with_name("harness-action-admission.toml")
FOLLOWUP_SDPO_CONFIG = CONFIG.with_name("harness-followup-sdpo.toml")
LAUNCHER = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_master_bootstrap_v1.sh"
ACTION_ADMISSION_LAUNCHER = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_action_admission_v1.sh"
ACTION_GATE_BATTERY = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_action_gate_battery_v1.sh"
MASTER_ADMISSION_LAUNCHER = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_master_admission_v1.sh"
ACTION_TRAIN_LAUNCHER = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_action_grpo_v1.sh"
CUMULATIVE_ACTION_TRAIN_LAUNCHER = (
    ROOT / "scripts" / "run_qwen35_27b_procedural_harness_send_followup_cumulative_grpo_v1.sh"
)
EVENT_CONTROL_TRAIN_LAUNCHER = (
    ROOT / "scripts" / "run_qwen35_27b_procedural_harness_send_followup_event_control_grpo_v1.sh"
)
FOLLOWUP_SDPO_LAUNCHER = ROOT / "scripts" / "run_qwen35_27b_procedural_harness_followup_sdpo_v1.sh"
FOLLOWUP_FEEDBACK_AUDIT = ROOT / "scripts" / "audit_procedural_harness_followup_feedback_v1.py"
PRIME_AGENT_RUNTIME_BUILDER = ROOT / "scripts" / "build_prime_agent_runtime_image_v1.sh"
PRIME_AGENT_RUNTIME_DOCKERFILE = CONFIG.with_name("prime-agent-runtime.Dockerfile")
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
    assert source.serve.pool.type == "static"
    assert source.serve.pool.num_workers == 1
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
    assert "record_causal_feedback = false" in admission_config
    assert "classify_curriculum_rung_admission" in admission_launcher
    assert "HARNESS_ACTION_ADMISSION_START_INDEX" in admission_launcher
    assert "HARNESS_ACTION_RECORD_CAUSAL_FEEDBACK" in admission_launcher
    assert "build_prime_agent_runtime_image_v1.sh" in admission_launcher
    assert (
        'image = "rlm-prime-agent-runtime:0.7.2-beta.495.1.97b994c-node22.19.0"'
        in admission_config
    )
    assert 'r"^start_index = [0-9]+$"' in admission_launcher
    assert "select_curriculum_rung_admission" in train_launcher
    assert "curriculum admission must contain eight error-free episodes" in selector
    assert 'status != "trainable"' in selector
    assert "HARNESS_ACTION_MODEL_PATH" in train_launcher
    assert "HARNESS_ACTION_MODEL_REPO" in train_launcher
    assert "HARNESS_ACTION_TRAIN_START_INDEX" in train_launcher
    assert "HARNESS_ACTION_TRAIN_COUNT" in train_launcher
    assert "HARNESS_ACTION_TRAIN_LR" in train_launcher
    assert "HARNESS_ACTION_BATCH_SIZE" in train_launcher
    assert "training start index must be non-negative" in train_launcher
    assert "training count must be positive" in train_launcher
    assert "r'^start_index = [0-9]+$'" in train_launcher
    assert "r'^count = [0-9]+$'" in train_launcher
    assert "training learning rate must be positive and finite" in train_launcher
    assert "training batch size must be a positive multiple of group size 8" in train_launcher
    assert "r'^lr = [^\\n]+$'" in train_launcher
    assert "r'^batch_size = [0-9]+$'" in train_launcher
    assert "oversampling_factor = 8 / batch_size" in train_launcher
    assert "r'^oversampling_factor = [^\\n]+$'" in train_launcher
    assert "refusing to launch while another GPU process is active" in train_launcher
    assert "bootstrap-shaped-grpo" not in train_launcher
    assert "HARNESS_ACTION_TRAIN_DRY_RUN" in train_launcher
    assert 'rl @ "$config" --model.name "$model_snapshot"' in train_launcher


def test_prime_agent_runtime_image_pins_the_episode_dependencies() -> None:
    builder = PRIME_AGENT_RUNTIME_BUILDER.read_text()
    dockerfile = PRIME_AGENT_RUNTIME_DOCKERFILE.read_text()

    assert "NODE_VERSION=22.19.0" in dockerfile
    assert "PRIME_AGENT_VERSION=0.7.2-beta.495.1.97b994c" in dockerfile
    assert "/var/tmp/vf-node" in dockerfile
    assert "/var/tmp/vf-prime-agent/${PRIME_AGENT_VERSION}" in dockerfile
    assert "PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=0" in dockerfile
    assert 'prime-agent" --version 2>&1)' in dockerfile
    assert 'docker image inspect "$image"' in builder
    assert 'docker run --rm "$image"' in builder
    assert "prime-agent --version 2>&1" in builder


def test_harness_action_gate_battery_is_disjoint_and_cumulative() -> None:
    launcher = ACTION_GATE_BATTERY.read_text()

    assert "HARNESS_ACTION_GATE_START_INDEX" in launcher
    assert "rungs=(atomic_state atomic_send atomic_followup)" in launcher
    assert "gate_start=$((start_index + offset * 1000))" in launcher
    assert "HARNESS_ACTION_ADMISSION_START_INDEX=$gate_start" in launcher
    assert '"$label-$rung-gate-r1"' in launcher


def test_send_followup_cumulative_grpo_preserves_the_prerequisite_in_the_batch() -> None:
    config = cli(RLConfig, args=["@", str(CUMULATIVE_ACTION_CONFIG), "--dry-run"])
    sources = {source.name: source for source in config.orchestrator.train.source}

    assert config.trainer.model.lora is None
    assert config.trainer.model.optimization_dtype == "bfloat16"
    assert config.trainer.optim.type == "adamw"
    assert config.trainer.optim.lr == 1.25e-7
    assert config.orchestrator.algo.type == "grpo"
    assert config.orchestrator.batch_size == 32
    assert config.orchestrator.group_size == 8
    assert config.orchestrator.oversampling_factor == 0.25
    assert config.orchestrator.batch_source_minimums == {
        "atomic-send-retention": 16,
        "atomic-followup-target": 16,
    }
    assert config.orchestrator.renderer.enable_thinking is True
    assert set(sources) == {"atomic-send-retention", "atomic-followup-target"}
    assert {source.ratio for source in sources.values()} == {1.0}
    assert {source.group_size for source in sources.values()} == {8}
    assert {source.env.taskset.curriculum_rung for source in sources.values()} == {"atomic_send", "atomic_followup"}
    assert len({source.env.taskset.start_index for source in sources.values()}) == 2
    assert all(source.algo is not None and source.algo.type == "grpo" for source in sources.values())
    assert all(source.env.taskset.task.reward_mode == "hard" for source in sources.values())
    assert all(source.serve.pool.type == "static" for source in sources.values())
    assert all(source.serve.pool.num_workers == 1 for source in sources.values())


def test_send_followup_cumulative_launcher_fails_closed_on_both_admissions() -> None:
    launcher = CUMULATIVE_ACTION_TRAIN_LAUNCHER.read_text()

    assert "HARNESS_CUMULATIVE_SEND_ADMISSION_SUMMARY" in launcher
    assert "HARNESS_CUMULATIVE_FOLLOWUP_ADMISSION_SUMMARY" in launcher
    assert 'for path, rung in zip(sys.argv[1:], ("atomic_send", "atomic_followup"), strict=True)' in launcher
    assert "select_curriculum_rung_admission(report, rung)" in launcher
    assert "send and follow-up training windows must be disjoint" in launcher
    assert "HARNESS_CUMULATIVE_SEND_START_INDEX" in launcher
    assert "HARNESS_CUMULATIVE_FOLLOWUP_START_INDEX" in launcher
    assert "HARNESS_CUMULATIVE_TRAIN_LR" in launcher
    assert "HARNESS_CUMULATIVE_BATCH_SIZE" in launcher
    assert "refusing to launch while another GPU process is active" in launcher
    assert "HARNESS_ACTION_MODEL_PATH is not a stable checkpoint" in launcher
    assert "HARNESS_CUMULATIVE_TRAIN_DRY_RUN" in launcher
    assert 'rl @ "$resolved_config" --model.name "$model_snapshot"' in launcher


def test_event_control_ramp_is_isolated_and_requires_measured_support() -> None:
    launcher = CUMULATIVE_ACTION_TRAIN_LAUNCHER.read_text()
    wrapper = EVENT_CONTROL_TRAIN_LAUNCHER.read_text()

    assert "HARNESS_CUMULATIVE_FOLLOWUP_REWARD_MODE" in launcher
    assert "HARNESS_CUMULATIVE_FOLLOWUP_SUPPORT_TRACES" in launcher
    assert "audit_procedural_harness_event_control_support_v1.py" in launcher
    assert 'reward_mode = "event_control"' in launcher
    assert "HARNESS_CUMULATIVE_FOLLOWUP_REWARD_MODE=event_control" in wrapper
    assert '"event-control-$label"' in wrapper
    assert "HARNESS_CUMULATIVE_SEND_START_INDEX:-1500000" in wrapper
    assert "HARNESS_CUMULATIVE_FOLLOWUP_START_INDEX:-1600000" in wrapper


def test_event_control_support_audit_requires_disconnected_informative_groups() -> None:
    traces = []
    for group_id, scores in (("a", (0.0, 0.5)), ("b", (0.0, 0.75))):
        for score in scores:
            traces.append(
                {
                    "ok": True,
                    "info": {"env_name": "atomic-followup-target", "group_id": group_id},
                    "metrics": {"event_control_progress": score},
                    "rewards": {"harness_score": {"score": 0.0}},
                }
            )

    report = summarize_support(traces, env_name="atomic-followup-target", group_size=2)
    validate_support(report, min_episodes=4, min_informative_groups=2)

    traces[0]["rewards"]["harness_score"]["score"] = 1.0
    report = summarize_support(traces, env_name="atomic-followup-target", group_size=2)
    try:
        validate_support(report, min_episodes=4, min_informative_groups=2)
    except ValueError as error:
        assert "hard-disconnected" in str(error)
    else:
        raise AssertionError("hard-connected support was accepted")


def test_followup_sdpo_bootstraps_only_the_typed_failed_transition() -> None:
    config = cli(RLConfig, args=["@", str(FOLLOWUP_SDPO_CONFIG), "--dry-run"])
    source = config.orchestrator.train.source[0]

    assert config.max_steps == 1
    assert config.max_train_batch_lead == 0
    assert config.trainer.model.lora is None
    assert config.trainer.model.optimization_dtype == "bfloat16"
    assert config.trainer.enable_token_export is True
    assert config.trainer.optim.type == "adamw"
    assert config.trainer.optim.lr == 2.5e-7
    assert config.trainer.sdpo_loss is not None
    assert config.trainer.sdpo_loss.teacher_regularization == "ema"
    assert config.orchestrator.batch_size == 16
    assert config.orchestrator.group_size == 1
    assert config.orchestrator.max_inflight_episodes == 8
    assert source.group_size == 1
    assert source.algo is not None and source.algo.type == "sdpo"
    assert source.algo.multi_turn_replay is True
    assert source.algo.require_explicit_feedback is True
    assert source.algo.required_feedback_contract_schema == "prime-agent/procedural-followup-feedback/v1"
    assert source.algo.filter.import_path == "procedural_harness_master_v1.taskset.keep_followup_feedback_response"
    assert source.env.taskset.curriculum_rung == "atomic_followup"
    assert source.env.taskset.record_causal_feedback is True
    assert source.env.taskset.task.reward_mode == "hard"
    assert source.env.agent.runtime.image == (
        "rlm-prime-agent-runtime:0.7.2-beta.495.1.97b994c-node22.19.0"
    )
    assert source.serve.pool.type == "static"
    assert source.serve.pool.num_workers == 1
    filters = {item.type: item for item in config.orchestrator.pre_batch_filters}
    assert filters["zero_advantage"].enforce is False


def test_followup_sdpo_launcher_requires_disconnection_and_feedback_audit() -> None:
    launcher = FOLLOWUP_SDPO_LAUNCHER.read_text()
    auditor = FOLLOWUP_FEEDBACK_AUDIT.read_text()

    assert 'classify_curriculum_rung_admission(admission, "atomic_followup")' in launcher
    assert '!= "disconnected"' in launcher
    assert "HARNESS_FOLLOWUP_FEEDBACK_AUDIT" in launcher
    assert 'audit.get("active_feedback_traces", 0) < 2' in launcher
    assert 'audit.get("structural_routing_verified") is not True' in launcher
    assert 'audit.get("routing_contract") != "one-mask-per-trainable-branch"' in launcher
    assert 'audit.get("answer_free") is not True' in launcher
    assert 'audit.get("failure_local") is not True' in launcher
    assert "HARNESS_ACTION_MODEL_PATH" in launcher
    assert "HARNESS_FOLLOWUP_TRAIN_START_INDEX" in launcher
    assert "HARNESS_FOLLOWUP_TRAIN_COUNT" in launcher
    assert "HARNESS_FOLLOWUP_TRAIN_LR" in launcher
    assert "HARNESS_FOLLOWUP_BATCH_SIZE" in launcher
    assert "refusing to launch while another GPU process is active" in launcher
    assert "HARNESS_FOLLOWUP_SDPO_DRY_RUN" in launcher
    assert "build_prime_agent_runtime_image_v1.sh" in launcher
    assert 'rl @ "$resolved_config" --model.name "$model_snapshot"' in launcher
    assert "keep_followup_feedback_response" in auditor
    assert "iter_trainable_branches" in auditor
    assert "feedback_contract_payload" in auditor
    assert "feedback leaks an answer value" in auditor


def test_action_gate_battery_bootstraps_and_health_checks_local_inference() -> None:
    battery = ACTION_GATE_BATTERY.read_text()
    admission = MASTER_ADMISSION_LAUNCHER.read_text()

    assert "run_qwen35_27b_prime_agent_mastery_baseline_v2.sh" in battery
    assert 'if [[ -z "$client_base_url" ]]' in battery
    assert "EVAL_DRIVER=" in battery
    assert "EVAL_EXPERIMENT_DIR=experiments/qwen35-27b-procedural-harness-master-v1" in battery
    assert "EVAL_CLIENT_HEALTH_URL" in admission
    assert "local evaluation endpoint is not healthy" in admission
