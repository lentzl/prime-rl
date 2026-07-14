import tomllib
from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.algorithm import SDPO_MAX_STUDENT_SUPPORT_TOPK, AlgorithmConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import BaseConfig, cli

# All config config classes
CONFIG_CLASSES = [
    RLConfig,
    TrainerConfig,
    SFTConfig,
    OrchestratorConfig,
    InferenceConfig,
]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`."""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


def is_eval_config(path: Path) -> bool:
    """vf-eval TOMLs live under configs but are not prime-rl entrypoint configs."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    return isinstance(data.get("eval"), list)


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path):
    """Tests that all config files can be loaded by at least one config class."""
    if is_eval_config(config_file):
        pytest.skip("vf-eval TOML files are not prime-rl entrypoint configs")

    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            cli(config_cls, args=["@", config_file.as_posix()])
            could_parse.append(True)
        except (ValidationError, ConfigFileError, SystemExit):
            could_parse.append(False)
    assert any(could_parse), f"No config class could be parsed from {config_file}"


class NestedConfig(BaseConfig):
    lr: float = 1e-4
    weight_decay: float = 0.01
    name: str = "default"


class VariantA(BaseModel):
    type: Literal["a"] = "a"
    alpha: float = 0.1
    shared: int = 1


class VariantB(BaseModel):
    type: Literal["b"] = "b"
    beta: float = 0.2
    shared: int = 1


VariantType = Annotated[VariantA | VariantB, Field(discriminator="type")]


class DummyConfig(BaseConfig):
    name: str = "experiment"
    seed: int = 42
    nested: NestedConfig = NestedConfig()
    variant: VariantType = VariantA()


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_defaults():
    """All defaults are applied when no TOML or CLI args are given."""
    config = cli(DummyConfig, args=[])
    assert config.name == "experiment"
    assert config.seed == 42
    assert config.nested.lr == 1e-4
    assert config.nested.weight_decay == 0.01
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.1


def test_toml_partial_nested_override(tmp_path):
    """Partially overriding a nested model preserves unset field defaults."""
    write_toml(tmp_path / "cfg.toml", {"nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.nested.lr == 3e-4
    assert config.nested.weight_decay == 0.01
    assert config.nested.name == "default"


def test_toml_discriminated_union_default_type(tmp_path):
    """Overriding a discriminated union field without 'type' uses the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"alpha": 0.9}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.9
    assert config.variant.shared == 1


def test_toml_discriminated_union_switch_variant(tmp_path):
    """Providing an explicit 'type' switches to that variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b"}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.2


def test_toml_discriminated_union_override_switch_variant(tmp_path):
    """Providing an explicit 'type' overrides the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b", "beta": 0.5}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.5


def test_cli_overrides_defaults():
    """CLI args override defaults."""
    config = cli(DummyConfig, args=["--name", "my-run", "--seed", "7"])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 1e-4


def test_toml_overrides_defaults(tmp_path):
    """TOML overrides defaults."""
    write_toml(tmp_path / "cfg.toml", {"name": "my-run", "seed": 7, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 3e-4


def test_cli_overrides_toml(tmp_path):
    """CLI args override TOML."""
    write_toml(tmp_path / "cfg.toml", {"seed": 1, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml"), "--seed", "99", "--nested.lr", "5e-5"])
    assert config.seed == 99
    assert config.nested.lr == 5e-5
    # TOML value not overridden by CLI should still be applied (not reverted to class default)
    assert config.nested.weight_decay == 0.01


def test_removed_fused_lm_head_chunk_size_field_is_rejected():
    with pytest.raises(ValidationError, match="fused_lm_head_chunk_size"):
        TrainerModelConfig.model_validate({"fused_lm_head_chunk_size": "auto"})


def test_env_algo_overrides_top_level():
    config = OrchestratorConfig.model_validate(
        {
            "renderer": {"name": "qwen3"},  # echo needs the renderer's role attribution
            "algo": {"type": "echo"},
            "train": {"env": [{"id": "a", "algo": {"type": "reward"}}, {"id": "b"}]},
        }
    )
    env_a, env_b = config.train.env
    # Env a sets its own algorithm; only env b inherits the top-level echo algorithm.
    assert env_a.algo is not None and env_a.algo.type == "reward"
    assert env_b.algo is not None and env_b.algo.type == "echo"

    # Resolved configs round-trip.
    dumped = config.model_dump(exclude_none=True)
    reloaded = OrchestratorConfig.model_validate(dumped)
    assert reloaded.train.env[0].algo is not None and reloaded.train.env[0].algo.type == "reward"


def test_orchestrator_resolve_env_config_rejects_unresolved_env_algorithm():
    config = OrchestratorConfig.model_validate({"train": {"env": [{"id": "a"}]}})
    config.train.env[0].algo = None

    with pytest.raises(ValueError, match="Each train env must have an algorithm config"):
        config.resolve_env_config()


def test_trainer_enable_token_export_cli_flag():
    assert not cli(TrainerConfig, args=[]).enable_token_export
    assert cli(TrainerConfig, args=["--enable-token-export"]).enable_token_export


def test_sdpo_defaults_pin_reference_knobs_and_ema_teacher_mode():
    trainer = TrainerConfig.model_validate({})
    assert trainer.sdpo_loss.full_logit_distillation
    assert trainer.sdpo_loss.distillation_topk == 100
    assert trainer.sdpo_loss.distillation_add_tail
    assert trainer.sdpo_loss.alpha == 0.5
    assert trainer.sdpo_loss.is_clip == 2.0
    assert trainer.sdpo_loss.rollout_is == "token"
    assert trainer.sdpo_loss.rollout_is_threshold == 2.0
    assert not trainer.sdpo_loss.rollout_is_batch_normalize
    assert trainer.sdpo_runtime.teacher_regularization == "live-policy"
    assert trainer.sdpo_runtime.teacher_update_rate == 0.05

    sdpo = TypeAdapter(AlgorithmConfig).validate_python({"type": "sdpo"})
    assert sdpo.teacher_regularization == "ema"
    assert sdpo.teacher_update_rate == 0.05
    assert sdpo.distillation_topk == 100
    assert sdpo.distillation_topk_support == "student"
    assert sdpo.preflight_export_timeout_s is None
    assert sdpo.success_reward_threshold == 0.5
    assert sdpo.successful_demonstration_selection == "batch_order"
    assert sdpo.dont_reprompt_on_self_success
    assert sdpo.remove_thinking_from_demonstration
    assert sdpo.include_environment_feedback
    assert sdpo.environment_feedback_only_without_solution
    assert sdpo.max_reprompt_len == 10240
    assert sdpo.reprompt_truncation == "right"
    assert (
        sdpo.template
        == "{question}{successful_solution_block}{feedback_block}\n\nCorrectly solve the original question."
    )
    assert sdpo.template_target == "first_user"
    assert sdpo.solution_template == "\nCorrect solution:\n\n{successful_previous_attempt}"
    assert (
        sdpo.feedback_template
        == "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"
    )
    assert sdpo.assistant_prefix == ""


@pytest.mark.parametrize(
    ("config_path", "teacher_regularization", "num_sdpo_teacher_gpus"),
    [
        ("configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml", "live-policy", 0),
        ("configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml", "ema", 1),
    ],
)
def test_sdpo_reference_smoke_configs_load_through_rl_config(
    config_path: str, teacher_regularization: str, num_sdpo_teacher_gpus: int
):
    config = cli(RLConfig, args=["@", config_path])

    assert config.orchestrator.algo.type == "sdpo"
    assert config.orchestrator.algo.teacher_regularization == teacher_regularization
    assert config.deployment.num_sdpo_teacher_gpus == num_sdpo_teacher_gpus
    assert config.uses_sdpo_student_support
    assert config.trainer.enable_token_export
    assert config.trainer.model.cp == 1
    assert config.trainer.model.fused_lm_head_token_chunk_size == "disabled"
    assert config.orchestrator.algo.distillation_topk == 100
    assert config.trainer.sdpo_loss.distillation_topk == 100
    assert config.trainer.sdpo_loss.rollout_is == "token"
    assert not config.trainer.sdpo_loss.rollout_is_batch_normalize
    assert config.inference is not None
    assert config.inference.vllm_extra["max_logprobs"] == 100


def test_sdpo_config_allows_reference_ablation_overrides():
    trainer = TrainerConfig.model_validate({"sdpo_loss": {"is_clip": None, "rollout_is": None}})
    assert trainer.sdpo_loss.is_clip is None
    assert trainer.sdpo_loss.rollout_is is None

    sdpo = TypeAdapter(AlgorithmConfig).validate_python(
        {
            "type": "sdpo",
            "teacher_regularization": "ema",
            "teacher_update_rate": 0.1,
            "distillation_topk_support": "student",
        }
    )
    assert sdpo.teacher_regularization == "ema"
    assert sdpo.teacher_update_rate == 0.1
    assert sdpo.distillation_topk_support == "student"


@pytest.mark.parametrize(
    "sdpo_loss",
    [
        {"distillation_topk": True},
        {"alpha": True},
        {"is_clip": True},
        {"rollout_is_threshold": True},
    ],
)
def test_sdpo_trainer_config_rejects_boolean_numeric_loss_knobs(sdpo_loss):
    with pytest.raises(ValidationError):
        TrainerConfig.model_validate({"sdpo_loss": sdpo_loss})


@pytest.mark.parametrize(
    "sdpo_loss",
    [
        {"full_logit_distillation": "yes"},
        {"distillation_add_tail": 1},
        {"rollout_is_batch_normalize": "yes"},
    ],
)
def test_sdpo_trainer_config_rejects_non_boolean_loss_behavior_knobs(sdpo_loss):
    with pytest.raises(ValidationError, match="must be a boolean"):
        TrainerConfig.model_validate({"sdpo_loss": sdpo_loss})


def test_sdpo_trainer_config_rejects_boolean_teacher_update_rate():
    with pytest.raises(ValidationError):
        TrainerConfig.model_validate({"sdpo_runtime": {"teacher_update_rate": True}})


@pytest.mark.parametrize(
    "algo",
    [
        {"type": "sdpo", "teacher_update_rate": True},
        {"type": "sdpo", "success_reward_threshold": True},
        {"type": "sdpo", "distillation_topk": True},
        {"type": "sdpo", "preflight_export_timeout_s": True},
        {"type": "sdpo", "max_reprompt_len": True},
        {"type": "sdpo", "max_concurrent": True},
    ],
)
def test_sdpo_algorithm_config_rejects_boolean_numeric_knobs(algo):
    with pytest.raises(ValidationError):
        TypeAdapter(AlgorithmConfig).validate_python(algo)


@pytest.mark.parametrize(
    "algo",
    [
        {"type": "sdpo", "dont_reprompt_on_self_success": "yes"},
        {"type": "sdpo", "remove_thinking_from_demonstration": 1},
        {"type": "sdpo", "include_environment_feedback": "yes"},
        {"type": "sdpo", "environment_feedback_only_without_solution": 0},
        {"type": "sdpo", "multi_turn": "on"},
    ],
)
def test_sdpo_algorithm_config_rejects_non_boolean_behavior_knobs(algo):
    with pytest.raises(ValidationError, match="must be a boolean"):
        TypeAdapter(AlgorithmConfig).validate_python(algo)


def test_sdpo_config_accepts_cli_boolean_strings_for_behavior_knobs(tmp_path):
    config_path = tmp_path / "sdpo.toml"
    write_toml(
        config_path,
        {
            "trainer": {"model": {"fused_lm_head_token_chunk_size": "disabled"}},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
                "train": {"env": [{"id": "reverse-text"}]},
            },
        },
    )

    config = cli(
        RLConfig,
        args=[
            "@",
            str(config_path),
            "--orchestrator.algo.multi_turn",
            "true",
            "--orchestrator.algo.include_environment_feedback",
            "false",
            "--trainer.sdpo_loss.distillation_add_tail",
            "false",
            "--trainer.sdpo_loss.rollout_is_batch_normalize",
            "false",
        ],
    )

    assert config.orchestrator.algo.multi_turn
    assert not config.orchestrator.algo.include_environment_feedback
    assert not config.trainer.sdpo_loss.distillation_add_tail
    assert not config.trainer.sdpo_loss.rollout_is_batch_normalize


def test_sdpo_config_rejects_template_with_unknown_field():
    with pytest.raises(ValidationError, match="sdpo template may only use named fields"):
        TypeAdapter(AlgorithmConfig).validate_python(
            {
                "type": "sdpo",
                "template": "{question}\n{unknown_feedback}",
            }
        )


@pytest.mark.parametrize(
    "template",
    [
        "{}",
        "{question.upper}",
        "{question[0]}",
    ],
)
def test_sdpo_config_rejects_template_field_traversal(template):
    with pytest.raises(ValidationError, match="sdpo template may only use named fields"):
        TypeAdapter(AlgorithmConfig).validate_python(
            {
                "type": "sdpo",
                "template": template,
            }
        )


def test_sdpo_config_rejects_nested_template_format_specs():
    with pytest.raises(ValidationError, match="template fields may not use format specs"):
        TypeAdapter(AlgorithmConfig).validate_python(
            {
                "type": "sdpo",
                "template": "{question:{hindsight_feedback}}",
            }
        )


def test_sdpo_config_rejects_template_conversion_flags():
    with pytest.raises(ValidationError, match="template fields may not use conversion flags"):
        TypeAdapter(AlgorithmConfig).validate_python(
            {
                "type": "sdpo",
                "template": "{question!r}",
            }
        )


def test_sdpo_config_rejects_template_format_specs():
    with pytest.raises(ValidationError, match="template fields may not use format specs"):
        TypeAdapter(AlgorithmConfig).validate_python(
            {
                "type": "sdpo",
                "template": "{question:>20}",
            }
        )


def test_sdpo_config_rejects_template_without_question_field():
    with pytest.raises(ValidationError, match=r"sdpo template must include the \{question\} field"):
        TypeAdapter(AlgorithmConfig).validate_python(
            {
                "type": "sdpo",
                "template": "{successful_solution_block}{feedback_block}",
            }
        )


def test_sdpo_config_rejects_template_without_hindsight_conditioning_field():
    with pytest.raises(ValidationError, match="hindsight conditioning field"):
        TypeAdapter(AlgorithmConfig).validate_python(
            {
                "type": "sdpo",
                "template": "{question}\n\nSolve again.",
            }
        )


def test_sdpo_config_accepts_template_with_raw_hindsight_feedback_field():
    sdpo = TypeAdapter(AlgorithmConfig).validate_python(
        {
            "type": "sdpo",
            "template": "{question}\n\nFeedback:\n{hindsight_feedback}",
        }
    )

    assert sdpo.template == "{question}\n\nFeedback:\n{hindsight_feedback}"


def test_sdpo_config_accepts_template_with_supported_feedback_fields():
    sdpo = TypeAdapter(AlgorithmConfig).validate_python(
        {
            "type": "sdpo",
            "template": (
                "{question}\n"
                "{successful_previous_rollout}\n"
                "{hindsight_feedback}\n"
                "{successful_solution_block}\n"
                "{feedback_block}"
            ),
        }
    )

    assert sdpo.template.startswith("{question}")


@pytest.mark.parametrize(
    ("field", "template", "message"),
    [
        ("solution_template", "{solution}", "solution_template may only use named fields"),
        (
            "solution_template",
            "{successful_previous_attempt!r}",
            "solution_template fields may not use conversion flags",
        ),
        ("solution_template", "{successful_previous_attempt:>20}", "solution_template fields may not use format specs"),
        ("solution_template", "Correct solution.", "solution_template must include"),
        ("feedback_template", "{feedback}", "feedback_template may only use named fields"),
        ("feedback_template", "{feedback_raw!r}", "feedback_template fields may not use conversion flags"),
        ("feedback_template", "{feedback_raw:>20}", "feedback_template fields may not use format specs"),
        ("feedback_template", "Feedback.", "feedback_template must include"),
    ],
)
def test_sdpo_config_validates_section_templates(field, template, message):
    with pytest.raises(ValidationError, match=message):
        TypeAdapter(AlgorithmConfig).validate_python(
            {
                "type": "sdpo",
                field: template,
            }
        )


def test_sdpo_config_accepts_custom_section_templates():
    sdpo = TypeAdapter(AlgorithmConfig).validate_python(
        {
            "type": "sdpo",
            "solution_template": "\nDemo:\n{successful_previous_attempt}",
            "feedback_template": "\nFeedback:\n{feedback_raw}",
        }
    )

    assert sdpo.solution_template == "\nDemo:\n{successful_previous_attempt}"
    assert sdpo.feedback_template == "\nFeedback:\n{feedback_raw}"


def test_sdpo_non_live_teacher_regularization_requires_policy_model_reference():
    with pytest.raises(ValidationError, match="internal self-distillation teacher mode"):
        TypeAdapter(AlgorithmConfig).validate_python(
            {
                "type": "sdpo",
                "model": {"name": "teacher-model", "base_url": ["http://teacher:8000/v1"]},
                "teacher_regularization": "ema",
            }
        )


def test_sdpo_live_policy_regularization_allows_external_teacher_ablation():
    sdpo = TypeAdapter(AlgorithmConfig).validate_python(
        {
            "type": "sdpo",
            "model": {"name": "teacher-model", "base_url": ["http://teacher:8000/v1"]},
            "teacher_regularization": "live-policy",
        }
    )

    assert sdpo.teacher_regularization == "live-policy"
    assert sdpo.model.name == "teacher-model"


def _rl_sdpo_student_support_config(**overrides):
    config = {
        "trainer": {},
        "orchestrator": {
            "renderer": {"name": "qwen3"},
            "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
            "train": {"env": [{"id": "reverse-text"}]},
        },
    }
    for key, value in overrides.items():
        if key == "trainer":
            config["trainer"].update(value)
        elif key == "orchestrator":
            config["orchestrator"].update(value)
        else:
            config[key] = value
    return config


def _assert_sdpo_reference_smoke_knobs(config, *, teacher_regularization: str):
    algo = config.orchestrator.algo
    assert algo.type == "sdpo"
    assert algo.model == "policy"
    assert algo.teacher_regularization == teacher_regularization
    assert algo.teacher_update_rate == 0.05
    assert algo.distillation_topk == 100
    assert algo.distillation_topk_support == "student"
    assert algo.preflight_export_timeout_s == 600
    assert algo.success_reward_threshold == 0.5
    assert algo.successful_demonstration_selection == "batch_order"
    assert algo.dont_reprompt_on_self_success
    assert algo.remove_thinking_from_demonstration
    assert algo.include_environment_feedback
    assert algo.environment_feedback_only_without_solution
    assert algo.max_reprompt_len == 10240
    assert algo.reprompt_truncation == "right"
    assert algo.template_target == "first_user"
    assert (
        algo.template
        == "{question}{successful_solution_block}{feedback_block}\n\nCorrectly solve the original question."
    )
    assert algo.solution_template == "\nCorrect solution:\n\n{successful_previous_attempt}"
    assert (
        algo.feedback_template
        == "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"
    )
    assert algo.assistant_prefix == ""
    assert not algo.multi_turn

    assert config.trainer.sdpo_loss.full_logit_distillation
    assert config.trainer.sdpo_loss.distillation_topk == 100
    assert config.trainer.sdpo_loss.distillation_add_tail
    assert config.trainer.sdpo_loss.alpha == 0.5
    assert config.trainer.sdpo_loss.is_clip == 2.0
    assert config.trainer.sdpo_loss.rollout_is == "token"
    assert config.trainer.sdpo_loss.rollout_is_threshold == 2.0
    assert not config.trainer.sdpo_loss.rollout_is_batch_normalize
    assert config.trainer.sdpo_runtime.teacher_regularization == teacher_regularization
    assert config.trainer.sdpo_runtime.teacher_update_rate == 0.05
    assert config.trainer.model.cp == 1


def test_rl_config_requires_provisioning_for_default_sdpo_ema_teacher():
    with pytest.raises(ValidationError, match="requires orchestrator.sdpo_teacher"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "sdpo"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


@pytest.mark.parametrize(
    "train",
    [
        {
            "sampling": {"temperature": 0.7},
            "env": [{"id": "reverse-text"}],
        },
        {
            "sampling": {"temperature": 1.0},
            "env": [{"id": "reverse-text", "sampling": {"temperature": 0.7}}],
        },
    ],
)
def test_rl_config_rejects_non_unit_sdpo_sampling_temperature(train):
    with pytest.raises(ValidationError, match=r"sdpo requires .*\.temperature=1\.0"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
                    "train": train,
                },
            }
        )


def test_rl_config_allows_sdpo_student_support_when_preflight_requirements_are_met():
    config = RLConfig.model_validate(_rl_sdpo_student_support_config())

    assert config.uses_sdpo_student_support
    assert config.trainer.enable_token_export
    assert config.trainer.model.fused_lm_head_token_chunk_size == "disabled"


def test_sdpo_live_policy_smoke_config_resolves_reference_runtime_knobs():
    config = cli(
        RLConfig,
        args=[
            "@",
            "configs/debug/algorithms/sdpo_huebotter_reference_smoke.toml",
            "--output-dir",
            "outputs/test-sdpo-smoke",
        ],
    )

    assert config.uses_sdpo_student_support
    assert not config.uses_sdpo_internal_teacher_regularization
    assert config.trainer.enable_token_export
    assert config.trainer.model.fused_lm_head_token_chunk_size == "disabled"
    assert config.inference is not None
    assert config.inference.vllm_extra["max_logprobs"] == 100
    _assert_sdpo_reference_smoke_knobs(config, teacher_regularization="live-policy")
    assert config.orchestrator.sdpo_teacher is None


def test_sdpo_ema_smoke_config_resolves_teacher_runtime_knobs():
    config = cli(
        RLConfig,
        args=[
            "@",
            "configs/debug/algorithms/sdpo_huebotter_reference_ema_smoke.toml",
            "--output-dir",
            "outputs/test-sdpo-ema-smoke",
        ],
    )

    assert config.uses_sdpo_student_support
    assert config.uses_sdpo_internal_teacher_regularization
    assert config.deployment.num_sdpo_teacher_gpus == 1
    assert config.trainer.enable_token_export
    assert config.trainer.model.fused_lm_head_token_chunk_size == "disabled"
    assert config.inference is not None
    assert config.inference.vllm_extra["max_logprobs"] == 100
    _assert_sdpo_reference_smoke_knobs(config, teacher_regularization="ema")
    assert config.orchestrator.sdpo_teacher is not None
    assert config.orchestrator.sdpo_teacher.name == config.orchestrator.model.name
    assert config.orchestrator.sdpo_teacher.client.base_url == ["http://localhost:8001/v1"]


def test_rl_config_auto_sets_managed_inference_max_logprobs_for_sdpo():
    config = RLConfig.model_validate(_rl_sdpo_student_support_config(inference={}))

    assert config.inference is not None
    assert config.inference.vllm_extra["max_logprobs"] == 100


def test_rl_config_preserves_larger_managed_inference_max_logprobs_for_sdpo():
    config = RLConfig.model_validate(_rl_sdpo_student_support_config(inference={"vllm_extra": {"max_logprobs": 128}}))

    assert config.inference is not None
    assert config.inference.vllm_extra["max_logprobs"] == 128


@pytest.mark.parametrize("max_logprobs", [20, True])
def test_rl_config_rejects_invalid_managed_inference_max_logprobs_for_sdpo(max_logprobs):
    with pytest.raises(ValidationError, match="inference\\.vllm_extra\\.max_logprobs"):
        RLConfig.model_validate(
            _rl_sdpo_student_support_config(inference={"vllm_extra": {"max_logprobs": max_logprobs}})
        )


def test_rl_config_keeps_teacher_support_ablation_off_preflight_path():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "algo": {
                    "type": "sdpo",
                    "teacher_regularization": "live-policy",
                    "distillation_topk_support": "teacher",
                },
                "train": {"env": [{"id": "reverse-text"}]},
            },
        }
    )

    assert not config.uses_sdpo_student_support
    assert not config.trainer.enable_token_export


def test_rl_config_rejects_sdpo_student_support_without_token_export():
    with pytest.raises(ValidationError, match="enable_token_export"):
        RLConfig.model_validate(_rl_sdpo_student_support_config(trainer={"enable_token_export": False}))


def test_rl_config_rejects_sdpo_student_support_when_topk_exceeds_vllm_candidate_limit():
    oversized_topk = SDPO_MAX_STUDENT_SUPPORT_TOPK + 1
    with pytest.raises(ValidationError, match="candidate-token scoring limit"):
        RLConfig.model_validate(
            _rl_sdpo_student_support_config(
                trainer={"sdpo_loss": {"distillation_topk": oversized_topk}},
                orchestrator={
                    "algo": {
                        "type": "sdpo",
                        "teacher_regularization": "live-policy",
                        "distillation_topk": oversized_topk,
                    }
                },
            )
        )


def test_rl_config_allows_sdpo_teacher_support_ablation_above_student_candidate_limit():
    oversized_topk = SDPO_MAX_STUDENT_SUPPORT_TOPK + 1
    config = RLConfig.model_validate(
        {
            "trainer": {"sdpo_loss": {"distillation_topk": oversized_topk}},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "algo": {
                    "type": "sdpo",
                    "teacher_regularization": "live-policy",
                    "distillation_topk": oversized_topk,
                    "distillation_topk_support": "teacher",
                },
                "train": {"env": [{"id": "reverse-text"}]},
            },
        }
    )

    assert config.orchestrator.algo.distillation_topk == oversized_topk
    assert not config.uses_sdpo_student_support


def test_rl_config_rejects_sdpo_student_support_without_trainer_logits():
    with pytest.raises(ValidationError, match="fused_lm_head_token_chunk_size"):
        RLConfig.model_validate(
            _rl_sdpo_student_support_config(
                trainer={"model": {"fused_lm_head_token_chunk_size": "auto"}},
            )
        )


def test_rl_config_rejects_sdpo_teacher_support_without_trainer_logits():
    with pytest.raises(ValidationError, match="fused_lm_head_token_chunk_size"):
        RLConfig.model_validate(
            {
                "trainer": {"model": {"fused_lm_head_token_chunk_size": "auto"}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {
                        "type": "sdpo",
                        "teacher_regularization": "live-policy",
                        "distillation_topk_support": "teacher",
                    },
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_sdpo_with_context_parallelism():
    with pytest.raises(ValidationError, match="trainer\\.model\\.cp > 1"):
        RLConfig.model_validate(_rl_sdpo_student_support_config(trainer={"model": {"cp": 2}}))


def test_rl_config_rejects_sdpo_rollout_is_batch_normalize_for_combined_runtime():
    with pytest.raises(ValidationError, match="rollout_is_batch_normalize"):
        RLConfig.model_validate(
            _rl_sdpo_student_support_config(
                trainer={"sdpo_loss": {"rollout_is_batch_normalize": True}},
            )
        )


def test_rl_config_rejects_sdpo_student_support_when_topk_mismatches_trainer():
    with pytest.raises(ValidationError, match="distillation_topk"):
        RLConfig.model_validate(
            _rl_sdpo_student_support_config(
                orchestrator={
                    "algo": {"type": "sdpo", "teacher_regularization": "live-policy", "distillation_topk": 64}
                },
            )
        )


def test_rl_config_rejects_sdpo_teacher_support_when_topk_mismatches_trainer():
    with pytest.raises(ValidationError, match="distillation_topk"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {
                        "type": "sdpo",
                        "teacher_regularization": "live-policy",
                        "distillation_topk": 64,
                        "distillation_topk_support": "teacher",
                    },
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


@pytest.mark.parametrize(
    "sdpo_loss",
    [
        {"full_logit_distillation": False, "distillation_topk": 100},
        {"full_logit_distillation": True, "distillation_topk": None},
    ],
)
def test_rl_config_rejects_sdpo_when_trainer_sdpo_loss_is_not_topk_full_logit(sdpo_loss):
    with pytest.raises(ValidationError, match="full-logit top-k"):
        RLConfig.model_validate(
            {
                "trainer": {"sdpo_loss": sdpo_loss},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {
                        "type": "sdpo",
                        "teacher_regularization": "live-policy",
                        "distillation_topk_support": "teacher",
                    },
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_detects_env_level_sdpo_student_support():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "algo": {"type": "reward"},
                "train": {
                    "env": [
                        {
                            "id": "reverse-text",
                            "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
                        }
                    ]
                },
            },
        }
    )

    assert config.uses_sdpo_student_support


def test_rl_config_detects_top_level_sdpo_inherited_by_explicit_train_envs():
    config = RLConfig.model_validate(
        {
            "trainer": {"model": {"fused_lm_head_token_chunk_size": "disabled"}},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
                "train": {"env": [{"id": "reverse-text"}, {"id": "reverse-text", "name": "second"}]},
            },
        }
    )

    assert [env.algo.type for env in config.orchestrator.train.env] == ["sdpo", "sdpo"]
    assert len(config.sdpo_algorithms) == 2
    assert config.uses_sdpo_student_support
    assert config.trainer.enable_token_export
    assert config.trainer.sdpo_runtime.teacher_regularization == "live-policy"
    assert config.trainer.sdpo_runtime.teacher_update_rate == 0.05


def test_rl_config_rejects_top_level_sdpo_inherited_by_train_envs_when_trainer_logits_are_fused():
    with pytest.raises(ValidationError, match="fused_lm_head_token_chunk_size"):
        RLConfig.model_validate(
            {
                "trainer": {"model": {"fused_lm_head_token_chunk_size": "auto"}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_detects_top_level_sdpo_without_train_env_for_validation():
    with pytest.raises(ValidationError, match="fused_lm_head_token_chunk_size"):
        RLConfig.model_validate(
            {
                "trainer": {"model": {"fused_lm_head_token_chunk_size": "auto"}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
                },
            }
        )


def test_rl_config_ignores_top_level_sdpo_when_all_train_envs_override_algorithm():
    config = RLConfig.model_validate(
        {
            "trainer": {"model": {"fused_lm_head_token_chunk_size": "auto"}},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "algo": {"type": "sdpo"},
                "train": {"env": [{"id": "reverse-text", "algo": {"type": "reward"}}]},
            },
        }
    )

    assert not config.sdpo_algorithms
    assert not config.uses_sdpo_student_support


def test_rl_config_rejects_env_level_sdpo_topk_mismatch_with_trainer():
    with pytest.raises(ValidationError, match="distillation_topk"):
        RLConfig.model_validate(
            {
                "trainer": {"sdpo_loss": {"distillation_topk": 100}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "reward"},
                    "train": {
                        "env": [
                            {
                                "id": "reverse-text",
                                "name": "sdpo-a",
                                "algo": {
                                    "type": "sdpo",
                                    "teacher_regularization": "live-policy",
                                    "distillation_topk": 100,
                                },
                            },
                            {
                                "id": "reverse-text",
                                "name": "sdpo-b",
                                "algo": {
                                    "type": "sdpo",
                                    "teacher_regularization": "live-policy",
                                    "distillation_topk": 64,
                                },
                            },
                        ]
                    },
                },
            }
        )


def test_rl_config_propagates_sdpo_teacher_knobs_to_trainer_runtime():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "algo": {"type": "sdpo", "teacher_regularization": "live-policy", "teacher_update_rate": 0.2},
                "train": {"env": [{"id": "reverse-text"}]},
            },
        }
    )

    assert config.trainer.sdpo_runtime.teacher_regularization == "live-policy"
    assert config.trainer.sdpo_runtime.teacher_update_rate == 0.2


def test_rl_config_rejects_non_live_trainer_sdpo_runtime_without_sdpo_algorithm():
    with pytest.raises(ValidationError, match="teacher_regularization != 'live-policy' requires an SDPO algorithm"):
        RLConfig.model_validate(
            {
                "trainer": {"sdpo_runtime": {"teacher_regularization": "ema"}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "grpo"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_trainer_sdpo_runtime_update_rate_without_sdpo_algorithm():
    with pytest.raises(ValidationError, match="teacher_update_rate requires an SDPO algorithm"):
        RLConfig.model_validate(
            {
                "trainer": {"sdpo_runtime": {"teacher_update_rate": 0.2}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "grpo"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_conflicting_trainer_sdpo_runtime_teacher_knobs():
    with pytest.raises(ValidationError, match="trainer.sdpo_runtime.teacher_update_rate conflicts"):
        RLConfig.model_validate(
            {
                "trainer": {"sdpo_runtime": {"teacher_update_rate": 0.2}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "sdpo", "teacher_update_rate": 0.1},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_conflicting_trainer_sdpo_runtime_teacher_regularization():
    with pytest.raises(ValidationError, match="trainer.sdpo_runtime.teacher_regularization conflicts"):
        RLConfig.model_validate(
            {
                "trainer": {"sdpo_runtime": {"teacher_regularization": "live-policy"}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8001/v1"]}},
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_conflicting_sdpo_algorithm_teacher_knobs():
    with pytest.raises(ValidationError, match="all sdpo algorithms.*teacher_update_rate"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "reward"},
                    "train": {
                        "env": [
                            {
                                "id": "reverse-text",
                                "name": "reverse-text-a",
                                "algo": {"type": "sdpo", "teacher_update_rate": 0.1},
                            },
                            {
                                "id": "reverse-text",
                                "name": "reverse-text-b",
                                "algo": {"type": "sdpo", "teacher_update_rate": 0.2},
                            },
                        ]
                    },
                },
            }
        )


def test_rl_config_rejects_conflicting_sdpo_algorithm_teacher_regularization():
    with pytest.raises(ValidationError, match="all sdpo algorithms.*teacher_regularization"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8001/v1"]}},
                    "algo": {"type": "reward"},
                    "train": {
                        "env": [
                            {
                                "id": "reverse-text",
                                "name": "reverse-text-live-policy",
                                "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
                            },
                            {
                                "id": "reverse-text",
                                "name": "reverse-text-ema",
                                "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                            },
                        ]
                    },
                },
            }
        )


def test_rl_config_rejects_conflicting_sdpo_algorithm_support_modes():
    with pytest.raises(ValidationError, match="all sdpo algorithms.*distillation_topk_support"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "reward"},
                    "train": {
                        "env": [
                            {
                                "id": "reverse-text",
                                "name": "reverse-text-student",
                                "algo": {"type": "sdpo", "distillation_topk_support": "student"},
                            },
                            {
                                "id": "reverse-text",
                                "name": "reverse-text-teacher",
                                "algo": {"type": "sdpo", "distillation_topk_support": "teacher"},
                            },
                        ]
                    },
                },
            }
        )


def test_rl_config_rejects_sdpo_ema_teacher_regularization_without_teacher_pool():
    with pytest.raises(ValidationError, match="orchestrator.sdpo_teacher"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_allows_sdpo_ema_teacher_regularization_with_distinct_teacher_pool():
    config = RLConfig.model_validate(
        {
            "trainer": {"model": {"name": "Qwen/Qwen3-4B-Instruct"}},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "model": {"name": "Qwen/Qwen3-4B-Instruct"},
                "sdpo_teacher": {"client": {"base_url": ["http://localhost:8001/v1"]}},
                "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                "train": {"env": [{"id": "reverse-text"}]},
            },
        }
    )

    assert config.uses_sdpo_internal_teacher_regularization
    assert config.orchestrator.sdpo_teacher is not None
    assert config.orchestrator.sdpo_teacher.name == "Qwen/Qwen3-4B-Instruct"
    assert config.trainer.sdpo_runtime.teacher_regularization == "ema"


def test_rl_config_rejects_sdpo_ema_teacher_model_mismatch():
    with pytest.raises(ValidationError, match="sdpo_teacher.name must match"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "model": {"name": "Qwen/Qwen3-4B-Instruct"},
                    "sdpo_teacher": {
                        "name": "Qwen/Qwen3-8B",
                        "client": {"base_url": ["http://localhost:8001/v1"]},
                    },
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_sdpo_ema_teacher_trust_remote_code_mismatch():
    with pytest.raises(ValidationError, match="sdpo_teacher.trust_remote_code must match"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "model": {"name": "Qwen/Qwen3-4B-Instruct", "trust_remote_code": True},
                    "sdpo_teacher": {
                        "trust_remote_code": False,
                        "client": {"base_url": ["http://localhost:8001/v1"]},
                    },
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_auto_sets_local_sdpo_teacher_endpoint_when_teacher_gpus_are_allocated():
    config = RLConfig.model_validate(
        {
            "trainer": {"model": {"name": "Qwen/Qwen3-4B-Instruct"}},
            "inference": {"model": {"name": "Qwen/Qwen3-4B-Instruct"}, "server": {"port": 8100}},
            "deployment": {"num_train_gpus": 1, "num_infer_gpus": 1, "num_sdpo_teacher_gpus": 1},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "model": {"name": "Qwen/Qwen3-4B-Instruct"},
                "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                "train": {"env": [{"id": "reverse-text"}]},
            },
        }
    )

    assert config.orchestrator.sdpo_teacher is not None
    assert config.orchestrator.model.client.base_url == ["http://localhost:8100/v1"]
    assert config.orchestrator.sdpo_teacher.name == "Qwen/Qwen3-4B-Instruct"
    assert config.orchestrator.sdpo_teacher.client.base_url == ["http://localhost:8101/v1"]
    assert config.uses_sdpo_internal_teacher_regularization


def test_rl_config_rejects_auto_sdpo_teacher_endpoint_port_overflow():
    with pytest.raises(ValidationError, match="inference.server.port \\+ 1 exceeds"):
        RLConfig.model_validate(
            {
                "trainer": {"model": {"name": "Qwen/Qwen3-4B-Instruct"}},
                "inference": {"model": {"name": "Qwen/Qwen3-4B-Instruct"}, "server": {"port": 65535}},
                "deployment": {"num_train_gpus": 1, "num_infer_gpus": 1, "num_sdpo_teacher_gpus": 1},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "model": {"name": "Qwen/Qwen3-4B-Instruct"},
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_auto_sets_local_sdpo_teacher_endpoint_for_env_level_sdpo():
    config = RLConfig.model_validate(
        {
            "trainer": {"model": {"name": "Qwen/Qwen3-4B-Instruct"}},
            "inference": {"model": {"name": "Qwen/Qwen3-4B-Instruct"}, "server": {"port": 8200}},
            "deployment": {"num_train_gpus": 1, "num_infer_gpus": 1, "num_sdpo_teacher_gpus": 1},
            "orchestrator": {
                "renderer": {"name": "qwen3"},
                "model": {"name": "Qwen/Qwen3-4B-Instruct"},
                "algo": {"type": "reward"},
                "train": {
                    "env": [
                        {
                            "id": "reverse-text",
                            "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                        }
                    ]
                },
            },
        }
    )

    assert config.uses_sdpo_internal_teacher_regularization
    assert config.trainer.sdpo_runtime.teacher_regularization == "ema"
    assert config.orchestrator.model.client.base_url == ["http://localhost:8200/v1"]
    assert config.orchestrator.sdpo_teacher is not None
    assert config.orchestrator.sdpo_teacher.client.base_url == ["http://localhost:8201/v1"]


def test_rl_config_rejects_sdpo_trust_region_until_reference_source_exists():
    with pytest.raises(ValidationError, match="trust-region.*not launchable"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8001/v1"]}},
                    "algo": {"type": "sdpo", "teacher_regularization": "trust-region"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_sdpo_teacher_endpoint_matching_auto_policy_port():
    with pytest.raises(ValidationError, match="not the policy endpoint"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "inference": {"server": {"port": 8100}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8100/v1"]}},
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_allocated_sdpo_teacher_gpus_without_non_live_sdpo():
    with pytest.raises(ValidationError, match="num_sdpo_teacher_gpus"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "inference": {},
                "deployment": {"num_train_gpus": 1, "num_infer_gpus": 1, "num_sdpo_teacher_gpus": 1},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_sdpo_teacher_endpoint_without_non_live_sdpo():
    with pytest.raises(ValidationError, match="orchestrator.sdpo_teacher requires an SDPO algorithm"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8001/v1"]}},
                    "algo": {"type": "sdpo", "teacher_regularization": "live-policy"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_sdpo_ema_teacher_regularization_on_policy_endpoint():
    with pytest.raises(ValidationError, match="not the policy endpoint"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8000/v1"]}},
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_sdpo_ema_teacher_regularization_with_nccl_broadcast():
    with pytest.raises(ValidationError, match="filesystem weight broadcast"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "weight_broadcast": {"type": "nccl"},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8001/v1"]}},
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_sdpo_ema_teacher_regularization_with_lora():
    with pytest.raises(ValidationError, match="not supported with LoRA"):
        RLConfig.model_validate(
            {
                "trainer": {"model": {"lora": {"rank": 8}}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8001/v1"]}},
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_sdpo_ema_teacher_regularization_with_multi_run_trainer():
    with pytest.raises(ValidationError, match="max_concurrent_runs"):
        RLConfig.model_validate(
            {
                "trainer": {"max_concurrent_runs": 2},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8001/v1"]}},
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_sdpo_ema_teacher_regularization_with_weights_only_checkpoints():
    with pytest.raises(ValidationError, match="weights_only"):
        RLConfig.model_validate(
            {
                "trainer": {"ckpt": {"weights_only": True}},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "sdpo_teacher": {"client": {"base_url": ["http://localhost:8001/v1"]}},
                    "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                    "train": {"env": [{"id": "reverse-text"}]},
                },
            }
        )


def test_rl_config_rejects_env_level_sdpo_ema_teacher_regularization_without_teacher_pool():
    with pytest.raises(ValidationError, match="orchestrator.sdpo_teacher"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "renderer": {"name": "qwen3"},
                    "algo": {"type": "reward"},
                    "train": {
                        "env": [
                            {
                                "id": "reverse-text",
                                "algo": {"type": "sdpo", "teacher_regularization": "ema"},
                            }
                        ]
                    },
                },
            }
        )


def test_single_node_auto_inference_client_dp_rank_count_matches_local_dp():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"parallel": {"tp": 1}},
            "deployment": {
                "type": "single_node",
                "gpus_per_node": 4,
                "num_train_gpus": 2,
                "num_infer_gpus": 2,
            },
        }
    )

    assert config.inference is not None
    assert config.inference.parallel.dp == 2
    assert config.orchestrator.model.client.dp_rank_count == 2


def test_single_node_rejects_inference_gpu_count_not_divisible_by_tensor_parallel_size():
    with pytest.raises(ValidationError, match="deployment.num_infer_gpus must be divisible"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {},
                "inference": {"parallel": {"tp": 2}},
                "deployment": {
                    "type": "single_node",
                    "gpus_per_node": 4,
                    "num_train_gpus": 1,
                    "num_infer_gpus": 3,
                },
            }
        )


def test_single_node_auto_inference_client_base_url_matches_custom_port_for_policy_sourced_env():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "inference": {"server": {"port": 8100}},
            "orchestrator": {
                "algo": {"type": "reward"},
                "train": {"env": [{"id": "reverse-text"}]},
            },
        }
    )

    assert config.orchestrator.model.client.base_url == ["http://localhost:8100/v1"]


def test_multi_node_auto_inference_client_dp_rank_count_uses_router_url():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"parallel": {"tp": 4}},
            "deployment": {
                "type": "multi_node",
                "gpus_per_node": 8,
                "num_train_nodes": 1,
                "num_infer_nodes": 2,
            },
            "slurm": {},
        }
    )

    assert config.inference is not None
    assert config.inference.data_parallel_size_local == 2
    assert config.inference.parallel.dp == 2
    assert config.orchestrator.model.client.dp_rank_count == 1


def test_orchestrator_vlm_requires_renderer():
    with pytest.raises(ValidationError, match="renderer"):
        OrchestratorConfig.model_validate(
            {
                "model": {
                    "name": "Qwen/Qwen3-VL-4B-Instruct",
                    "vlm": {
                        "vision_encoder_attr": "model.visual",
                        "language_model_attr": "model.language_model",
                    },
                },
                "renderer": None,
            }
        )

    config = OrchestratorConfig.model_validate(
        {
            "model": {
                "name": "Qwen/Qwen3-VL-4B-Instruct",
                "vlm": {
                    "vision_encoder_attr": "model.visual",
                    "language_model_attr": "model.language_model",
                },
            },
        }
    )

    assert config.renderer is not None


def test_selective_activation_checkpointing_requires_custom_impl():
    with pytest.raises(ValidationError, match="Selective activation checkpointing requires model.impl='custom'"):
        TrainerModelConfig.model_validate({"impl": "hf", "ac": {"mode": "selective"}})


def test_shared_model_name_propagates_to_subconfigs():
    model_name = "PrimeIntellect/test-model"
    config = RLConfig.model_validate(
        {
            "model": {"name": model_name},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {},
        }
    )
    assert config.trainer.model.name == model_name
    assert config.orchestrator.model.name == model_name
    assert config.inference is not None and config.inference.model.name == model_name
    assert config.trainer.tokenizer.name == model_name
    assert config.orchestrator.tokenizer.name == model_name


def test_shared_tokenizer_propagates_when_subconfigs_unset():
    config = RLConfig.model_validate(
        {
            "model": {"name": "my-model"},
            "tokenizer": {"name": "my-tokenizer"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.tokenizer.name == "my-tokenizer"
    assert config.orchestrator.tokenizer.name == "my-tokenizer"


def test_shared_and_sub_tokenizer_name_conflict_raises():
    """Setting tokenizer.name in both [tokenizer] and [trainer.tokenizer]
    is a config conflict — the sub-config would silently win, and any later
    CLI override of [tokenizer].name would silently no-op for the trainer."""
    with pytest.raises(ValidationError, match=r"tokenizer.name.*trainer.tokenizer.name"):
        RLConfig.model_validate(
            {
                "model": {"name": "my-model"},
                "tokenizer": {"name": "shared-tok"},
                "trainer": {"tokenizer": {"name": "trainer-tok"}},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )


def test_tokenizer_name_falls_back_to_model_name_when_unset():
    config = RLConfig.model_validate(
        {
            "model": {"name": "my-model"},
            "tokenizer": {"trust_remote_code": True},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.tokenizer.name == "my-model"
    assert config.orchestrator.tokenizer.name == "my-model"
    assert config.trainer.tokenizer.trust_remote_code is True
    assert config.orchestrator.tokenizer.trust_remote_code is True


def test_explicit_subconfig_tokenizer_name_survives_shared_model_propagation():
    """Regression: shared ``[model] name = "M"`` must propagate model names but
    must NOT clobber an explicit ``[orchestrator.tokenizer] name = "T"``.

    This is the case that the old RL-level ``auto_setup_tokenizer`` fix-up got
    wrong: it unconditionally re-derived ``orchestrator.tokenizer.name`` from
    ``orchestrator.model.name`` after propagation, silently overriding
    the user's explicit value. The ``mode="before"`` ``auto_setup_shared_configs``
    propagator fixes this because it propagates the model name into the raw
    dict before sub-configs are built, so ``OrchestratorConfig``'s own
    ``auto_setup_tokenizer`` (mode=after) sees the resolved name *and* the
    explicit user-set tokenizer name, and the ``fill``-if-absent semantic
    leaves the explicit value alone.
    """
    config = RLConfig.model_validate(
        {
            "model": {"name": "M"},
            "trainer": {},
            "orchestrator": {
                "renderer": {"name": "default"},
                "tokenizer": {"name": "explicit-orch-tok"},
            },
        }
    )
    # Shared model.name reached every sub-config that didn't override it.
    assert config.trainer.model.name == "M"
    assert config.orchestrator.model.name == "M"
    # Trainer didn't specify a tokenizer, so it falls back to the propagated model name.
    assert config.trainer.tokenizer.name == "M"
    # Orchestrator's explicit tokenizer name survived.
    assert config.orchestrator.tokenizer.name == "explicit-orch-tok"


def test_tokenizer_chat_template_mismatch_raises():
    with pytest.raises(ValidationError, match="chat_template"):
        RLConfig.model_validate(
            {
                "trainer": {"tokenizer": {"chat_template": "A"}},
                "orchestrator": {"renderer": {"name": "default"}, "tokenizer": {"chat_template": "B"}},
            }
        )


def test_shared_seq_len_propagates_to_subconfigs():
    config = RLConfig.model_validate(
        {
            "seq_len": 4096,
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.model.seq_len == 4096
    assert config.orchestrator.seq_len == 4096


def test_shared_and_sub_seq_len_conflict_raises():
    """Setting seq_len at the shared level and on a sub-config is a conflict —
    forces the user to pick one place to express the value rather than
    relying on the silent 'sub wins' rule."""
    with pytest.raises(ValidationError, match=r"seq_len.*trainer.model.seq_len"):
        RLConfig.model_validate(
            {
                "seq_len": 4096,
                "trainer": {"model": {"seq_len": 8192}},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )


def test_shared_and_sub_model_name_conflict_raises():
    """Setting model.name at the shared level and on a sub-config is a conflict."""
    with pytest.raises(ValidationError, match=r"model.name.*trainer.model.name"):
        RLConfig.model_validate(
            {
                "model": {"name": "X"},
                "trainer": {"model": {"name": "Y"}},
                "orchestrator": {"renderer": {"name": "default"}},
            }
        )


def test_shared_and_sub_max_steps_conflict_raises():
    """Top-level scalar shared fields also participate in the mutex check."""
    with pytest.raises(ValidationError, match=r"max_steps.*orchestrator.max_steps"):
        RLConfig.model_validate(
            {
                "max_steps": 100,
                "trainer": {},
                "orchestrator": {"renderer": {"name": "default"}, "max_steps": 200},
            }
        )


def test_trainer_chat_template_cascades_to_inference():
    """``[trainer.tokenizer] chat_template`` set directly (no shared
    ``[tokenizer] chat_template``) must still reach
    ``inference.model.chat_template`` so vLLM's ``--chat-template`` is wired
    up. Regression: the original ``auto_setup_tokenizer`` cascaded this; the
    refactored propagator must keep doing it."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {"tokenizer": {"chat_template": "TPL"}},
            "orchestrator": {"renderer": {"name": "default"}, "tokenizer": {"chat_template": "TPL"}},
            "inference": {},
        }
    )
    assert config.trainer.tokenizer.chat_template == "TPL"
    assert config.orchestrator.tokenizer.chat_template == "TPL"
    assert config.inference is not None
    assert config.inference.model.chat_template == "TPL"


def test_shared_wandb_fields_propagate_to_subconfigs():
    """Every ``SharedWandbConfig`` leaf (project, entity, name, group, tags,
    offline) propagates to both trainer.wandb and orchestrator.wandb. Regression
    for a miss in the inline propagator."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "wandb": {
                "project": "shared-proj",
                "entity": "shared-entity",
                "name": "shared-name",
                "group": "shared-group",
                "tags": ["a", "b"],
                "offline": False,
            },
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    for component in (config.trainer.wandb, config.orchestrator.wandb):
        assert component is not None
        assert component.project == "shared-proj"
        assert component.entity == "shared-entity"
        assert component.name == "shared-name"
        assert component.group == "shared-group"
        assert component.tags == ["a", "b"]
        assert component.offline is False


def test_empty_shared_ckpt_block_does_not_conflict_with_subconfig_ckpt():
    """An empty shared [ckpt] block is a presence-only signal, not a field
    setting — it should not conflict with a non-empty [trainer.ckpt]."""
    config = RLConfig.model_validate(
        {
            "ckpt": {},  # empty block, no field set
            "trainer": {"ckpt": {"interval": 50}},
            "orchestrator": {"renderer": {"name": "default"}, "ckpt": {"interval": 50}},
        }
    )
    assert config.trainer.ckpt is not None
    assert config.trainer.ckpt.interval == 50


def test_shared_and_subconfig_disjoint_fields_coexist():
    """Per-field mutex only forbids conflicts on the SAME field — disjoint
    fields in [model] vs [trainer.model] are fine."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {"model": {"impl": "custom"}},
            "orchestrator": {"renderer": {"name": "default"}},
        }
    )
    assert config.trainer.model.name == "Qwen/Qwen3-0.6B"
    assert config.trainer.model.impl == "custom"


def test_shared_output_dir_propagates_through_cli(tmp_path):
    """Shared output_dir from CLI reaches sub-configs even when tyro constructs sub-configs before the before-validator."""
    toml_path = tmp_path / "cfg.toml"
    write_toml(
        toml_path,
        {
            "max_steps": 1,
            "seq_len": 128,
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "trainer": {},
            "orchestrator": {"batch_size": 16, "group_size": 1},
            "inference": {},
        },
    )
    shared_out = tmp_path / "shared"
    config = cli(RLConfig, args=["@", str(toml_path), "--output-dir", str(shared_out)])
    assert config.trainer.output_dir == shared_out
    assert config.orchestrator.output_dir == shared_out / "run_default"


def test_orchestrator_renderer_auto_rejects_unmapped_model():
    """Default ``renderer`` (AutoRendererConfig) must reject models not in MODEL_RENDERER_MAP."""
    with pytest.raises(ValidationError, match="silently fall back to DefaultRenderer"):
        OrchestratorConfig.model_validate({"model": {"name": "not-a-real-org/not-a-real-model"}})


def test_orchestrator_renderer_auto_accepts_mapped_model():
    """The default Qwen model is in MODEL_RENDERER_MAP and should validate cleanly."""
    config = OrchestratorConfig.model_validate({"model": {"name": "Qwen/Qwen3-0.6B"}})
    assert config.renderer is not None
    assert config.renderer.name == "auto"


def test_orchestrator_explicit_renderer_skips_unmapped_check():
    """Explicit renderer.name bypasses the auto-resolution check — user opted in."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"name": "qwen3"},
        }
    )
    assert config.renderer is not None
    assert config.renderer.name == "qwen3"


def test_orchestrator_renderer_none_rejected():
    """A renderer is required (training is renderer-only): the non-optional type rejects None."""
    with pytest.raises(ValidationError, match="renderer"):
        OrchestratorConfig.model_validate(
            {
                "model": {"name": "not-a-real-org/not-a-real-model"},
                "renderer": None,
            }
        )


def test_orchestrator_explicit_default_renderer_with_unmapped_model():
    """renderer.name='default' is an explicit opt-in to DefaultRenderer and must pass."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"name": "default", "tool_parser": "qwen3"},
        }
    )
    assert config.renderer is not None
    assert config.renderer.name == "default"
    assert config.renderer.tool_parser == "qwen3"


def test_shared_model_name_resolves_inference_parsers():
    """Shared [model] name must reach inference.model BEFORE ModelConfig's after-validator
    runs auto_resolve_parsers — i.e. the parsers resolve from the propagated name, not
    from an empty default.
    """
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {},
        }
    )
    assert config.inference is not None
    assert config.inference.model.name == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert config.inference.model.tool_call_parser == "qwen3_coder"


def test_explicit_inference_parser_wins_over_auto():
    """Explicit inference.model.tool_call_parser is preserved even when the shared model
    name would otherwise auto-resolve to something else."""
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
            "inference": {"model": {"tool_call_parser": "hermes"}},
        }
    )
    assert config.inference is not None
    assert config.inference.model.tool_call_parser == "hermes"
