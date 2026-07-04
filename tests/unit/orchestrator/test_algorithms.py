import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pydantic
import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage

import prime_rl.orchestrator.algo as algo_module
from prime_rl.configs.algorithm import AlgorithmConfig, FrozenModelConfig
from prime_rl.orchestrator.algo import (
    EchoAlgorithm,
    OPSDAlgorithm,
    SDPOAlgorithm,
    build_algorithm,
    stamp_advantages,
    stamp_loss_routing,
)
from prime_rl.orchestrator.sdpo_student_support import SDPOStudentSupportRecord, hydrate_student_support_from_records
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout, RolloutView
from prime_rl.transport.types import TrainingSample

FROZEN = {"name": "org/ref-model", "base_url": ["http://ref:8001/v1"]}

_ALGO = pydantic.TypeAdapter(AlgorithmConfig)


def _build(**kwargs) -> AlgorithmConfig:
    """Validate an algorithm config — ``algo.type`` is the discriminator (the
    bundle IS the algorithm)."""
    return _ALGO.validate_python(kwargs)


def _ref_kind(ref):
    """Collapse a resolved reference to a comparable marker."""
    return "frozen" if isinstance(ref, FrozenModelConfig) else ref


@pytest.mark.parametrize(
    ("algorithm_type", "teacher", "source", "model_ref", "action_loss_type"),
    [
        ("grpo", None, "policy", None, "rl"),
        ("max_rl", None, "policy", None, "rl"),
        ("opd", FROZEN, "policy", "frozen", "ref_kl"),
        ("sft", FROZEN, "frozen", None, "ce"),
        ("opsd", None, "policy", "policy", "ref_kl"),
        ("sdpo", None, "policy", "policy", "sdpo"),
        ("echo", None, "policy", None, "rl"),
    ],
)
def test_type_defaults_are_the_vetted_algorithms(algorithm_type, teacher, source, model_ref, action_loss_type):
    kwargs = {"type": algorithm_type}
    if teacher is not None:
        kwargs["teacher"] = teacher
    algo = _build(**kwargs)
    assert _ref_kind(algo.sampling.source) == source
    assert algo.type == algorithm_type
    assert _ref_kind(getattr(algo, "model", None)) == model_ref
    assert algo.action_loss_type == action_loss_type


def test_build_algorithm_dispatches_sdpo_runtime_contract():
    config = _build(type="sdpo")
    algo = build_algorithm(config, MagicMock(), MagicMock())

    assert isinstance(algo, SDPOAlgorithm)
    assert algo.action_loss_type == config.action_loss_type == "sdpo"
    assert algo.model_role == "teacher"
    assert algo.batch_preflight_name == "sdpo_student_support"
    assert algo.distillation_topk_support == "student"


def test_build_algorithm_rejects_runtime_config_loss_type_mismatch(monkeypatch):
    class _WrongSDPORuntime:
        action_loss_type = "rl"

    monkeypatch.setitem(algo_module.ALGORITHM_CLASSES, "sdpo", _WrongSDPORuntime)

    with pytest.raises(ValueError, match="action_loss_type mismatch.*sdpo.*runtime='rl'.*config='sdpo'"):
        build_algorithm(_build(type="sdpo"), MagicMock(), MagicMock())


def test_sdpo_direct_construction_requires_renderer():
    with pytest.raises(ValueError, match="requires the renderer"):
        SDPOAlgorithm(_build(type="sdpo"), MagicMock(), None)


def test_sdpo_score_batch_empty_batch_does_not_require_teacher_clients():
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), MagicMock())
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[])

    asyncio.run(algo.score_batch([]))


def test_sdpo_score_batch_rejects_missing_teacher_train_clients():
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), _CaptureRenderer(token_ids=[101, 102]))
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(RuntimeError, match="no train clients configured"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_echo_roles_replace_the_default_table():
    algo = _build(type="echo", roles={"user": {"alpha": 0.5}})
    assert algo.type == "echo"
    assert algo.roles.user.alpha == 0.5
    # Setting any role replaces the whole table: the tool default is gone
    assert algo.roles.tool is None


def test_echo_defaults_to_tool_bodies():
    algo = _build(type="echo")
    assert algo.roles.tool.alpha == 0.1
    assert algo.roles.system is None
    assert algo.roles.user is None
    assert algo.roles.assistant is None


def test_echo_roles_require_at_least_one():
    with pytest.raises(ValueError, match="at least one role"):
        _build(type="echo", roles={})


def test_opd_requires_teacher():
    with pytest.raises(ValueError, match="needs a teacher"):
        _build(type="opd")


def test_sft_requires_teacher():
    with pytest.raises(ValueError, match="needs a teacher to sample rollouts from"):
        _build(type="sft")


def test_teacher_folds_into_model():
    algo = _build(type="opd", teacher=FROZEN)
    assert isinstance(algo.model, FrozenModelConfig)
    assert algo.model.name == "org/ref-model"


def test_teacher_without_target_errors():
    with pytest.raises(ValueError, match="references no model"):
        _build(type="grpo", teacher=FROZEN)


def test_teacher_redundant_but_consistent_is_accepted():
    algo = _build(type="opd", teacher=FROZEN, model=FROZEN)
    assert isinstance(algo.model, FrozenModelConfig)


def test_opd_rejects_policy():
    with pytest.raises(ValueError, match="degenerate"):
        _build(type="opd", model="policy")


def test_rl_loss_type_incompatible_with_frozen_sampling():
    with pytest.raises(ValueError, match="sampling.source is a frozen model"):
        _build(type="grpo", sampling={"source": FROZEN})


def test_sdpo_policy_teacher_uses_live_policy_pool():
    policy_pool = MagicMock()
    algo = SDPOAlgorithm(_build(type="sdpo", model="policy"), policy_pool, _CaptureRenderer(token_ids=[101, 102]))

    asyncio.run(algo.setup())

    assert algo.teacher_pool is policy_pool
    assert algo.connected_pools == []


def test_sdpo_ema_teacher_regularization_requires_live_teacher_pool():
    algo = SDPOAlgorithm(
        _build(type="sdpo", teacher_regularization="ema"),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )

    with pytest.raises(NotImplementedError, match="sdpo_teacher"):
        asyncio.run(algo.setup())


def test_sdpo_ema_teacher_regularization_uses_live_teacher_pool():
    policy_pool = MagicMock()
    teacher_pool = MagicMock()
    algo = SDPOAlgorithm(
        _build(type="sdpo", teacher_regularization="ema"),
        policy_pool,
        _CaptureRenderer(token_ids=[101, 102]),
        live_pools={"sdpo_teacher": teacher_pool},
    )

    asyncio.run(algo.setup())

    assert algo.teacher_pool is teacher_pool
    assert algo.connected_pools == []


def test_sdpo_score_batch_requires_setup_before_scoring():
    algo = SDPOAlgorithm(
        _build(type="sdpo"),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )

    with pytest.raises(RuntimeError, match="Algorithm\\.setup\\(\\) must run before score_batch"):
        asyncio.run(algo.score_batch([]))


def test_sdpo_accepts_student_topk_support_mode():
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk_support="student"),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )

    assert algo.distillation_topk_support == "student"


def test_sdpo_student_support_declares_batch_preflight_only_for_weighted_samples():
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk_support="student"),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )

    assert algo.batch_preflight_name == "sdpo_student_support"
    assert not algo.needs_batch_preflight(
        [SimpleNamespace(samples=[SimpleNamespace(sdpo_weights=None, mask=[False, False])])]
    )
    assert algo.needs_batch_preflight(
        [SimpleNamespace(samples=[SimpleNamespace(sdpo_weights=None, mask=[False, True])])]
    )
    assert not algo.needs_batch_preflight([SimpleNamespace(samples=[SimpleNamespace(sdpo_weights=[0.0, 0.0])])])
    assert algo.needs_batch_preflight([SimpleNamespace(samples=[SimpleNamespace(sdpo_weights=[0.0, 1.0])])])


def test_sdpo_student_support_preflight_signature_names_export_contract():
    default_timeout = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )
    smoke_timeout = SDPOAlgorithm(
        _build(
            type="sdpo",
            distillation_topk=2,
            distillation_topk_support="student",
            preflight_export_timeout_s=600,
        ),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )

    assert default_timeout.batch_preflight_signature() == (
        "sdpo_student_support",
        "support=student",
        "sample_ids=strict",
        2,
        None,
    )
    assert smoke_timeout.batch_preflight_signature() == (
        "sdpo_student_support",
        "support=student",
        "sample_ids=strict",
        2,
        600,
    )
    assert default_timeout.batch_preflight_signature() != smoke_timeout.batch_preflight_signature()


def test_sdpo_teacher_support_stays_off_preflight_path():
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk_support="teacher"),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )

    assert not algo.needs_batch_preflight([SimpleNamespace(samples=[SimpleNamespace(sdpo_weights=[1.0])])])


# --------------------------------------------------------------------------
# Routing / advantage stamping over the FLAT TrainingSample data model.
#
# A sample is a single flat token sequence: ``mask`` marks the trainable
# (model-sampled) tokens; the streams (rl/ce/ref_kl/advantages) are all
# full-length-N (= len(token_ids)), 0.0 on non-trainable positions.
# --------------------------------------------------------------------------


def _make_sample(ce_weights: list[float] | None = None) -> TrainingSample:
    # 2 prompt tokens (mask False), then a 4-token completion with one
    # env-provided observation token (position 4, mask False) interleaved.
    return TrainingSample(
        token_ids=[1, 2, 3, 4, 5, 6],
        mask=[False, False, True, True, False, True],
        logprobs=[0.0, 0.0, -0.1, -0.2, 0.0, -0.3],
        temperatures=[],
        env_name="test-env",
        ce_weights=ce_weights,
    )


def test_stamp_loss_routing_uniform_rl():
    sample = _make_sample()
    stamp_loss_routing(sample, "rl")
    # Hot path: absent streams mean rl weight 1.0 on the loss mask
    assert sample.rl_weights is None
    assert sample.ce_weights is None
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_ref_kl_action():
    sample = _make_sample()
    stamp_loss_routing(sample, "ref_kl")
    # Action tokens (mask True) feed the ref_kl component; rl is off
    assert sample.rl_weights == [0.0] * 6
    assert sample.ref_kl_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert sample.ce_weights is None


def test_stamp_loss_routing_ce_action():
    sample = _make_sample()
    stamp_loss_routing(sample, "ce")
    assert sample.rl_weights == [0.0] * 6
    assert sample.ce_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_keeps_algorithm_written_ce_stream():
    # Echo writes ce_weights directly at group time (observation at position
    # 4, outside the loss mask); rl routing must not clobber it — the rl
    # component still ships no streams (hot path).
    sample = _make_sample(ce_weights=[0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    stamp_loss_routing(sample, "rl")
    assert sample.rl_weights is None
    assert sample.ce_weights == [0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_merges_action_weights_into_ce_stream():
    # A ce-action algorithm that also weighted observation tokens: action
    # tokens merge into the existing stream instead of replacing it.
    sample = _make_sample(ce_weights=[0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    stamp_loss_routing(sample, "ce")
    assert sample.rl_weights == [0.0] * 6
    assert sample.ce_weights == [0.0, 0.0, 1.0, 1.0, 0.1, 1.0]
    assert sample.ref_kl_weights is None


def _make_rollout(
    samples: list[TrainingSample],
    advantages: list[float] | None = None,
) -> Rollout:
    rollout = Rollout(task=vf.Task(idx=0, prompt=None), nodes=[], rewards={}, env_name="test-env")
    rollout.samples = samples
    rollout.advantages = advantages
    return rollout


def test_stamp_advantages_full_length_stream():
    # The advantage stream is full-length-N: 0.0 on prompt + non-trainable
    # positions, the rl credit on trainable (mask True) tokens.
    rollout = _make_rollout([_make_sample()], advantages=[0.0, 0.0, 0.5, -0.5, 0.0, 1.0])
    stamp_advantages(rollout)
    assert rollout.samples[0].advantages == [0.0, 0.0, 0.5, -0.5, 0.0, 1.0]


def test_stamp_advantages_slices_across_samples():
    samples = [_make_sample(), _make_sample()]
    rollout = _make_rollout(samples, advantages=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    stamp_advantages(rollout)
    assert rollout.samples[0].advantages == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert rollout.samples[1].advantages == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


def test_stamp_advantages_no_credit_ships_none():
    rollout = _make_rollout([_make_sample()])
    stamp_advantages(rollout)
    assert rollout.samples[0].advantages is None


def test_stamp_advantages_rejects_misaligned():
    rollout = _make_rollout([_make_sample()], advantages=[0.5])
    with pytest.raises(ValueError, match="align"):
        stamp_advantages(rollout)


def test_assign_advantages_scalar_broadcasts_over_mask():
    rollout = _make_rollout([_make_sample()])
    RolloutView(rollout).assign_advantages(1.0)
    assert rollout.advantages == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_assign_advantages_list_rejects_misaligned():
    rollout = _make_rollout([_make_sample()])
    with pytest.raises(ValueError, match="align"):
        RolloutView(rollout).assign_advantages([0.5])


# --------------------------------------------------------------------------
# Echo: weighted CE on env-provided observation tokens of later turns.
#
# Provenance is structural under v1 — within a branch, the non-sampled nodes
# that follow the first sampled (model) node are the env-provided observations
# (tool output, user feedback). Each such node's token span gets its message
# role's weight; the initial prompt (before the first response) is excluded.
# --------------------------------------------------------------------------


def _echo_algorithm(roles: dict | None = None, filter_fn=None) -> EchoAlgorithm:
    kwargs: dict = {"type": "echo"}
    if roles is not None:
        kwargs["roles"] = roles
    algo = EchoAlgorithm(_build(**kwargs), MagicMock(), MagicMock())
    algo.filter_fn = filter_fn
    return algo


def _node(message, *, parent, sampled, token_ids, logprobs=None) -> MessageNode:
    return MessageNode(
        parent=parent,
        message=message,
        sampled=sampled,
        token_ids=token_ids,
        mask=[sampled] * len(token_ids),
        logprobs=logprobs if logprobs is not None else ([0.0] * len(token_ids) if sampled else []),
    )


def _two_turn_rollout(observation_role: str = "tool", *, reward: float = 1.0) -> Rollout:
    """A single linear branch: user prompt, an assistant response, an
    env-provided observation (tool output / user feedback), then a second
    assistant response. Tokens: prompt [1,2], action [3,4], observation
    [5,6], action [7,8]."""
    if observation_role == "tool":
        obs_message = ToolMessage(tool_call_id="t", content="T")
    else:
        obs_message = UserMessage(content="feedback")
    nodes = [
        _node(UserMessage(content="U"), parent=None, sampled=False, token_ids=[1, 2]),
        _node(AssistantMessage(content="A"), parent=0, sampled=True, token_ids=[3, 4], logprobs=[-0.1, -0.2]),
        _node(obs_message, parent=1, sampled=False, token_ids=[5, 6]),
        _node(AssistantMessage(content="A2"), parent=2, sampled=True, token_ids=[7, 8], logprobs=[-0.3, -0.4]),
    ]
    rollout = Rollout(task=vf.Task(idx=0, prompt=None), nodes=nodes, rewards={"r": reward}, env_name="test-env")
    rollout.samples = trace_to_samples(rollout, env_name="test-env")
    return rollout


def _branching_rollout() -> Rollout:
    """Two trainable branches sharing the initial user prompt."""
    nodes = [
        _node(UserMessage(content="U"), parent=None, sampled=False, token_ids=[1, 2]),
        _node(AssistantMessage(content="left-1"), parent=0, sampled=True, token_ids=[3, 4], logprobs=[-0.1, -0.2]),
        _node(ToolMessage(tool_call_id="t", content="left-tool"), parent=1, sampled=False, token_ids=[5]),
        _node(AssistantMessage(content="left-2"), parent=2, sampled=True, token_ids=[6], logprobs=[-0.3]),
        _node(AssistantMessage(content="right-1"), parent=0, sampled=True, token_ids=[7, 8], logprobs=[-0.4, -0.5]),
    ]
    rollout = Rollout(task=vf.Task(idx=0, prompt=None), nodes=nodes, rewards={"r": 1.0}, env_name="test-env")
    rollout.samples = trace_to_samples(rollout, env_name="test-env")
    return rollout


def test_echo_weights_observations_by_role():
    # The observation node [5,6] follows the first sampled node, so it is
    # weighted; the initial prompt [1,2] precedes it and is excluded.
    rollout = _two_turn_rollout()
    algo = _echo_algorithm()  # the default table: tool bodies at 0.1
    asyncio.run(algo.score_rollout(RolloutView(rollout)))
    sample = rollout.samples[0]
    assert sample.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sample.mask == [False, False, True, True, False, False, True, True]
    # [3,4] step-1 action, [5,6] observation (weighted), [7,8] step-2 action
    assert sample.ce_weights == [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0]

    # A user-feedback observation under a role table that weights users.
    rollout = _two_turn_rollout(observation_role="user")
    algo = _echo_algorithm(roles={"tool": {"alpha": 0.1}, "user": {"alpha": 0.05}})
    asyncio.run(algo.score_rollout(RolloutView(rollout)))
    assert rollout.samples[0].ce_weights == [0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0]

    # A role not in the table leaves the observation unweighted: no ce stream.
    rollout = _two_turn_rollout(observation_role="user")
    algo = _echo_algorithm()  # tool only
    asyncio.run(algo.score_rollout(RolloutView(rollout)))
    assert rollout.samples[0].ce_weights is None


def test_echo_filter_narrows_selection():
    # A per-branch keep-mask drops observation position 5 (the second tool
    # token), narrowing the role selection.
    def keep_drop_one(trace):
        # One keep-mask per trainable branch, spanning that branch's tokens.
        return [[True, True, True, True, True, False, True, True]]

    rollout = _two_turn_rollout()
    algo = _echo_algorithm(filter_fn=keep_drop_one)
    asyncio.run(algo.score_rollout(RolloutView(rollout)))
    assert rollout.samples[0].ce_weights == [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]

    # Shape violations fail loudly: wrong branch count, wrong per-branch length.
    rollout = _two_turn_rollout()
    with pytest.raises(ValueError, match="per trainable branch"):
        asyncio.run(_echo_algorithm(filter_fn=lambda trace: []).score_rollout(RolloutView(rollout)))
    rollout = _two_turn_rollout()
    with pytest.raises(ValueError, match="span the branch's tokens"):
        asyncio.run(_echo_algorithm(filter_fn=lambda trace: [[True] * 6]).score_rollout(RolloutView(rollout)))


# --------------------------------------------------------------------------
# OPSD reference scoring. These tests characterize the generic hook that SDPO-
# style feedback-conditioned self-teaching should build on before adding any
# new trainer path.
# --------------------------------------------------------------------------


class _CaptureRenderer:
    def __init__(self, token_ids: list[int] | list[list[int]], text_ids: list[int] | None = None):
        self.token_ids = token_ids
        self.text_ids = text_ids or [901, 902]
        self.calls: list[dict] = []
        self.text_calls: list[str] = []

    def render_ids(self, messages, *, add_generation_prompt):
        self.calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt})
        if self.token_ids and isinstance(self.token_ids[0], list):
            return self.token_ids[len(self.calls) - 1]
        return self.token_ids

    def encode_text(self, text):
        self.text_calls.append(text)
        return self.text_ids


class _AssistantPrefillRenderer(_CaptureRenderer):
    def __init__(self, prefill_ids: list[int]):
        super().__init__(token_ids=[])
        self.prefill_ids = prefill_ids
        self.prefill_calls: list[dict] = []

    def render_assistant_prefill_ids(self, messages, text):
        self.prefill_calls.append({"messages": messages, "text": text})
        return self.prefill_ids


class _RenderAssistantPrefillRenderer(_CaptureRenderer):
    def __init__(self):
        super().__init__(token_ids=[101, 102], text_ids=[901, 902])
        self.render_calls: list[dict] = []

    def render(self, messages, *, add_generation_prompt):
        self.render_calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt})
        return SimpleNamespace(
            token_ids=[101, 102, 201, 202, 999, 1000],
            message_indices=[0, 0, 1, 1, 1, 1],
            sampled_mask=[False, False, True, True, True, False],
        )

    def get_stop_token_ids(self):
        return [999]


class _MalformedAssistantPrefillRenderer(_CaptureRenderer):
    def __init__(self, *, close_token: int = 999, sampled_mask: list[bool] | None = None):
        super().__init__(token_ids=[101, 102], text_ids=[901, 902])
        self.close_token = close_token
        self.sampled_mask = sampled_mask or [False, False, True, True, True, False]

    def render(self, messages, *, add_generation_prompt):
        return SimpleNamespace(
            token_ids=[101, 102, 201, 202, self.close_token, 1000],
            message_indices=[0, 0, 1, 1, 1, 1],
            sampled_mask=self.sampled_mask,
        )

    def get_stop_token_ids(self):
        return [999]


def _single_turn_rollout(
    info: dict | None = None,
    *,
    reward: float = 1.0,
    assistant_content: str = "Attempt",
    feedback_content: str | None = None,
    group_id: uuid.UUID | None = None,
) -> Rollout:
    nodes = [
        _node(UserMessage(content="Solve the task."), parent=None, sampled=False, token_ids=[1, 2]),
        _node(
            AssistantMessage(content=assistant_content),
            parent=0,
            sampled=True,
            token_ids=[3, 4],
            logprobs=[-0.3, -0.4],
        ),
    ]
    if feedback_content is not None:
        nodes.append(_node(UserMessage(content=feedback_content), parent=1, sampled=False, token_ids=[5, 6]))
    rollout = Rollout(
        task=vf.Task(idx=0, prompt=None),
        nodes=nodes,
        rewards={"r": reward},
        info=info or {},
        env_name="test-env",
    )
    if group_id is not None:
        rollout.group_id = group_id
    rollout.samples = trace_to_samples(rollout, env_name="test-env")
    return rollout


def _single_turn_rollout_with_nontrainable_leaf_before_sampled_leaf(*, reward: float = 0.0) -> Rollout:
    nodes = [
        _node(UserMessage(content="Solve the task."), parent=None, sampled=False, token_ids=[1, 2]),
        _node(UserMessage(content="Side branch."), parent=0, sampled=False, token_ids=[5, 6]),
        _node(
            AssistantMessage(content="Attempt"),
            parent=0,
            sampled=True,
            token_ids=[3, 4],
            logprobs=[-0.3, -0.4],
        ),
        _node(UserMessage(content="Use the sampled branch feedback."), parent=2, sampled=False, token_ids=[7, 8]),
    ]
    rollout = Rollout(
        task=vf.Task(idx=0, prompt=None),
        nodes=nodes,
        rewards={"r": reward},
        env_name="test-env",
    )
    rollout.samples = trace_to_samples(rollout, env_name="test-env")
    return rollout


def test_opsd_builds_demo_conditioned_reference_prefix():
    renderer = _CaptureRenderer(token_ids=[101, 102, 103])
    algo = OPSDAlgorithm(
        _build(
            type="opsd",
            demo_key="feedback",
            template="{question}\n\nFeedback:\n{demonstration}\n\nTry again.",
        ),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout(info={"feedback": "The first attempt missed the invariant."})

    assert algo._ref_prefix_ids(RolloutView(rollout)) == [101, 102, 103]
    assert renderer.calls == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": ("Solve the task.\n\nFeedback:\nThe first attempt missed the invariant.\n\nTry again."),
                }
            ],
            "add_generation_prompt": True,
        }
    ]


def test_opsd_template_can_use_rollout_info_fields():
    renderer = _CaptureRenderer(token_ids=[101, 102, 103])
    algo = OPSDAlgorithm(
        _build(
            type="opsd",
            template=("{question}\n\nFeedback: {judge_feedback}\nCorrection:\n{demonstration}"),
        ),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout(
        info={
            "demonstration": "Inspect state, store the active id, then answer.",
            "judge_feedback": "The sampled attempt guessed without inspecting state.",
        }
    )

    assert algo._ref_prefix_ids(RolloutView(rollout)) == [101, 102, 103]
    assert renderer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": (
                "Solve the task.\n\n"
                "Feedback: The sampled attempt guessed without inspecting state.\n"
                "Correction:\n"
                "Inspect state, store the active id, then answer."
            ),
        }
    ]


def test_opsd_scores_completion_under_reference_prefix(monkeypatch):
    async def fake_prefill_logprobs(client, model_name, token_ids):
        assert model_name == "policy-model"
        assert token_ids == [101, 102, 3, 4]
        return [-0.1, -0.2, -0.7, -0.8]

    monkeypatch.setattr("prime_rl.orchestrator.algo.opsd.compute_prefill_logprobs", fake_prefill_logprobs)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = OPSDAlgorithm(_build(type="opsd"), MagicMock(), renderer)
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(info={"demonstration": "Use the invariant."})

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].ref_logprobs == [0.0, 0.0, -0.7, -0.8]


def test_opsd_multi_turn_is_opt_in():
    algo = _build(type="opsd", multi_turn=True)
    assert algo.multi_turn is True


def test_opsd_can_condition_the_first_user_message():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = OPSDAlgorithm(
        _build(
            type="opsd",
            template_target="first_user",
            template="{question}\n\nFeedback:\n{demonstration}",
        ),
        MagicMock(),
        renderer,
    )
    messages = [
        {"role": "user", "content": "Original task"},
        {"role": "assistant", "content": "Attempt"},
        {"role": "user", "content": "Runtime feedback"},
    ]

    assert algo._demo_conditioned_prefix_ids(messages, "Use the invariant.", "test-env") == [101, 102]
    assert renderer.calls[0]["messages"] == [
        {"role": "user", "content": "Original task\n\nFeedback:\nUse the invariant."},
        {"role": "assistant", "content": "Attempt"},
        {"role": "user", "content": "Runtime feedback"},
    ]


def test_opsd_rejects_multi_turn_rollouts():
    algo = OPSDAlgorithm(_build(type="opsd"), MagicMock(), _CaptureRenderer(token_ids=[101, 102]))
    rollout = _two_turn_rollout()
    rollout.info["demonstration"] = "Use the feedback from the first turn."

    with pytest.raises(ValueError, match="single-step trajectories only"):
        algo._ref_prefix_ids(RolloutView(rollout))


def test_opsd_scores_multi_turn_completion_segments_when_enabled(monkeypatch):
    calls = []

    async def fake_prefill_logprobs(client, model_name, token_ids):
        calls.append(token_ids)
        assert model_name == "policy-model"
        if len(calls) == 1:
            assert token_ids == [101, 102, 3, 4]
            return [-0.1, -0.2, -0.7, -0.8]
        assert token_ids == [201, 202, 7, 8]
        return [-0.1, -0.2, -0.9, -1.0]

    monkeypatch.setattr("prime_rl.orchestrator.algo.opsd.compute_prefill_logprobs", fake_prefill_logprobs)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = OPSDAlgorithm(
        _build(
            type="opsd",
            multi_turn=True,
            template="{question}\n\nFeedback:\n{demonstration}",
        ),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout()
    rollout.info["demonstration"] = "First response missed the useful state."

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert calls == [[101, 102, 3, 4], [201, 202, 7, 8]]
    assert rollout.samples[0].ref_logprobs == [0.0, 0.0, -0.7, -0.8, 0.0, 0.0, -0.9, -1.0]
    assert renderer.calls == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "U\n\nFeedback:\nFirst response missed the useful state.",
                }
            ],
            "add_generation_prompt": True,
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "U\n\nFeedback:\nFirst response missed the useful state.",
                },
                {"role": "assistant", "content": "A"},
                {"role": "tool", "content": "T", "tool_call_id": "t"},
            ],
            "add_generation_prompt": True,
        },
    ]


def test_opsd_multi_turn_preserves_user_feedback_when_conditioning_first_user(monkeypatch):
    async def fake_prefill_logprobs(client, model_name, token_ids):
        if token_ids == [101, 102, 3, 4]:
            return [-0.1, -0.2, -0.7, -0.8]
        assert token_ids == [201, 202, 7, 8]
        return [-0.1, -0.2, -0.9, -1.0]

    monkeypatch.setattr("prime_rl.orchestrator.algo.opsd.compute_prefill_logprobs", fake_prefill_logprobs)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = OPSDAlgorithm(
        _build(
            type="opsd",
            multi_turn=True,
            template_target="first_user",
            template="{question}\n\nFeedback:\n{demonstration}",
        ),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout(observation_role="user")
    rollout.info["demonstration"] = "The first attempt missed the useful state."

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].ref_logprobs == [0.0, 0.0, -0.7, -0.8, 0.0, 0.0, -0.9, -1.0]
    assert renderer.calls[1]["messages"] == [
        {
            "role": "user",
            "content": "U\n\nFeedback:\nThe first attempt missed the useful state.",
        },
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "feedback"},
    ]


def test_opsd_multi_turn_can_use_branch_local_hindsight_feedback(monkeypatch):
    async def fake_prefill_logprobs(client, model_name, token_ids):
        if token_ids == [101, 102, 3, 4]:
            return [-0.1, -0.2, -0.7, -0.8]
        assert token_ids == [201, 202, 7, 8]
        return [-0.1, -0.2, -0.9, -1.0]

    monkeypatch.setattr("prime_rl.orchestrator.algo.opsd.compute_prefill_logprobs", fake_prefill_logprobs)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = OPSDAlgorithm(
        _build(
            type="opsd",
            multi_turn=True,
            template=(
                "{question}\n\n"
                "Hindsight feedback for the sampled turn:\n{hindsight_feedback}\n\n"
                "Correction:\n{demonstration}"
            ),
        ),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout()
    rollout.info["demonstration"] = "Use the tool feedback before repairing."

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert renderer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": (
                "U\n\n"
                "Hindsight feedback for the sampled turn:\n"
                "tool: T\n\n"
                "Correction:\n"
                "Use the tool feedback before repairing."
            ),
        }
    ]
    assert renderer.calls[1]["messages"] == [
        {
            "role": "user",
            "content": (
                "U\n\n"
                "Hindsight feedback for the sampled turn:\n\n\n"
                "Correction:\n"
                "Use the tool feedback before repairing."
            ),
        },
        {"role": "assistant", "content": "A"},
        {"role": "tool", "content": "T", "tool_call_id": "t"},
    ]


def test_opsd_scores_each_multi_turn_branch_independently(monkeypatch):
    async def fake_prefill_logprobs(client, model_name, token_ids):
        assert model_name == "policy-model"
        completion = token_ids[-2:] if token_ids[-2:] in ([3, 4], [7, 8]) else token_ids[-1:]
        if completion == [3, 4]:
            return [0.0] * (len(token_ids) - 2) + [-0.7, -0.8]
        if completion == [6]:
            return [0.0] * (len(token_ids) - 1) + [-0.6]
        assert completion == [7, 8]
        return [0.0] * (len(token_ids) - 2) + [-0.9, -1.0]

    monkeypatch.setattr("prime_rl.orchestrator.algo.opsd.compute_prefill_logprobs", fake_prefill_logprobs)
    renderer = _CaptureRenderer(token_ids=[[101], [201], [301]])
    algo = OPSDAlgorithm(_build(type="opsd", multi_turn=True), MagicMock(), renderer)
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _branching_rollout()
    rollout.info["demonstration"] = "Use branch-local feedback."

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert len(rollout.samples) == 2
    ref_logprobs_by_tokens = {tuple(sample.token_ids): sample.ref_logprobs for sample in rollout.samples}
    assert ref_logprobs_by_tokens[(1, 2, 3, 4, 5, 6)] == [0.0, 0.0, -0.7, -0.8, 0.0, -0.6]
    assert ref_logprobs_by_tokens[(1, 2, 7, 8)] == [0.0, 0.0, -0.9, -1.0]
    assert len(renderer.calls) == 3


def test_opsd_multi_turn_rejects_branch_sample_token_misalignment():
    algo = OPSDAlgorithm(_build(type="opsd", multi_turn=True), MagicMock(), _CaptureRenderer(token_ids=[101, 102]))
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout()
    rollout.info["demonstration"] = "Use the branch context."
    rollout.samples[0].token_ids = [*rollout.samples[0].token_ids, 9]

    with pytest.raises(ValueError, match="sample tokens to align"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_opsd_multi_turn_rejects_branch_sample_mask_misalignment():
    algo = OPSDAlgorithm(_build(type="opsd", multi_turn=True), MagicMock(), _CaptureRenderer(token_ids=[101, 102]))
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout()
    rollout.info["demonstration"] = "Use the branch context."
    rollout.samples[0].mask = rollout.samples[0].mask[:-1]

    with pytest.raises(ValueError, match="sample mask to align"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


# --------------------------------------------------------------------------
# SDPO top-k teacher scoring.
# --------------------------------------------------------------------------


def test_sdpo_routes_action_tokens_to_sdpo_component():
    sample = _make_sample()
    stamp_loss_routing(sample, "sdpo")

    assert sample.rl_weights == [0.0] * 6
    assert sample.sdpo_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert sample.ref_kl_weights is None


def test_sdpo_scores_completion_topk_under_feedback_conditioned_prefix(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        assert model_name == "policy-model"
        assert topk == 2
        assert token_ids == [101, 102, 3, 4]
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert rollout.samples[0].sdpo_topk_token_ids == [[0, 0], [0, 0], [30, 31], [40, 41], [0, 0], [0, 0]]
    assert rollout.samples[0].sdpo_topk_logprobs == [
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.5, -1.5],
        [-0.4, -1.4],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    assert renderer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": (
                "Solve the task.\n"
                "The following is feedback from your unsuccessful earlier attempt:\n\n"
                "user: The attempt ignored the visible state.\n\n"
                "Correctly solve the original question."
            ),
        }
    ]
    assert renderer.text_calls == []


def test_sdpo_teacher_support_rejects_empty_teacher_prefix_before_prefill(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("empty SDPO teacher prefixes must fail before teacher top-k scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")

    with pytest.raises(ValueError, match="teacher reprompt rendered to an empty token prefix"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_scoring_preserves_existing_component_weights(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].sdpo_weights = [0.0, 0.0, 0.5, 0.75, 0.0, 0.0]

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_weights == [0.0, 0.0, 0.5, 0.75, 0.0, 0.0]


def test_sdpo_teacher_support_rejects_precomputed_rollout_is_weights_before_teacher_scoring(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("stale rollout-IS weights must fail before teacher scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].sdpo_rollout_is_weights = [0.0, 0.0, 0.7, 0.8, 0.0, 0.0]

    with pytest.raises(ValueError, match="rollout-IS weights"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (1.0, "sdpo_weights must be a list"),
        ([0.0], "sdpo_weights length"),
        ([0.0, 0.0, True, 1.0, 0.0, 0.0], "finite numeric"),
        ([0.0, 0.0, float("nan"), 1.0, 0.0, 0.0], "finite numeric"),
        ([0.0, 0.0, -0.5, 1.0, 0.0, 0.0], "non-negative"),
        ([1.0, 0.0, 1.0, 1.0, 0.0, 0.0], "zero outside sampled tokens"),
    ],
)
def test_sdpo_rejects_malformed_component_weights_before_teacher_scoring(monkeypatch, weights, message):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("malformed sdpo_weights must fail before teacher scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].sdpo_weights = weights

    with pytest.raises(ValueError, match=message):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("token_ids", (1, 2, 3, 4, 5, 6), "sample token_ids must be a list"),
        ("mask", (False, False, True, True, False, False), "sample mask must be a list"),
    ],
)
def test_sdpo_weight_validation_rejects_non_list_sample_streams(field, value, message):
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].sdpo_weights = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    setattr(rollout.samples[0], field, value)

    with pytest.raises(ValueError, match=message):
        algo._validate_sdpo_weights(rollout.samples[0])


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        ([False, False, True, True, False], "sample mask length"),
        ([False, False, 1, True, False, False], "sample mask must contain booleans"),
    ],
)
def test_sdpo_weight_validation_rejects_malformed_sample_mask(mask, message):
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].sdpo_weights = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    rollout.samples[0].mask = mask

    with pytest.raises(ValueError, match=message):
        algo._validate_sdpo_weights(rollout.samples[0])


def test_sdpo_rejects_branch_sample_mask_value_misalignment(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("misaligned masks must fail before teacher scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].mask = [False, False, True, True, True, False]

    with pytest.raises(ValueError, match="sample mask values to match"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_single_turn_rollout_with_multiple_samples_before_teacher_scoring(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("malformed single-turn sample count must fail before teacher scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples.append(rollout.samples[0])

    with pytest.raises(ValueError, match="exactly one sample for single-turn rollout"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_preflight_rejects_single_turn_rollout_with_multiple_samples():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use the feedback.")
    rollout.samples.append(rollout.samples[0])

    with pytest.raises(ValueError, match="exactly one sample for single-turn rollout"):
        algo.select_batch_preflight_samples(
            [RolloutView(rollout)],
            samples=rollout.samples,
        )


def test_sdpo_single_turn_scores_the_trainable_branch_not_the_first_leaf(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        assert model_name == "policy-model"
        assert topk == 2
        assert token_ids == [101, 102, 3, 4]
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout_with_nontrainable_leaf_before_sampled_leaf()

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert len(rollout.samples) == 1
    assert rollout.samples[0].token_ids == [1, 2, 3, 4, 7, 8]
    assert rollout.samples[0].sdpo_topk_token_ids == [[0, 0], [0, 0], [30, 31], [40, 41], [0, 0], [0, 0]]
    assert renderer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": (
                "Solve the task.\n"
                "The following is feedback from your unsuccessful earlier attempt:\n\n"
                "user: Use the sampled branch feedback.\n\n"
                "Correctly solve the original question."
            ),
        }
    ]


def test_sdpo_student_support_preflight_selects_the_trainable_branch_not_the_first_leaf():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout_with_nontrainable_leaf_before_sampled_leaf()
    sample = rollout.samples[0]
    sample.sdpo_weights = None

    selected = algo.select_batch_preflight_samples(
        [RolloutView(rollout)],
        samples=[sample],
    )

    assert selected == [sample]
    assert sample.sdpo_weights == [1.0 if trains else 0.0 for trains in sample.mask]


def test_sdpo_teacher_support_scores_only_nonzero_sdpo_target_positions(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        assert token_ids == [101, 102, 3, 4]
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].sdpo_weights = [0.0, 0.0, 0.0, 0.75, 0.0, 0.0]

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_topk_token_ids == [[0, 0], [0, 0], [0, 0], [40, 41], [0, 0], [0, 0]]
    assert rollout.samples[0].sdpo_topk_logprobs == [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.4, -1.4],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        ([2, 2], "target positions must be unique"),
        ([-1], "target position outside token range"),
        ([6], "target position outside token range"),
        ([True], "target positions must be integer token indices"),
    ],
)
def test_sdpo_target_positions_reject_malformed_position_lists(positions, message):
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )
    sample = _make_sample()
    sample.sdpo_weights = [0.0, 0.0, 0.5, 0.75, 0.0, 0.0]

    with pytest.raises(ValueError, match=message):
        algo._sdpo_target_positions(sample, positions)


def test_sdpo_scatter_rejects_target_position_drift_before_writeback():
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )
    sample = _make_sample()
    sample.sdpo_weights = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="scatter target positions changed before writeback"):
        algo._scatter_topk(
            sample,
            [[30, 31], [40, 41]],
            [[-0.5, -1.5], [-0.4, -1.4]],
            positions=[2, 3],
        )


def test_sdpo_prefers_raw_rollout_info_feedback_for_single_turn(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(
        info={"feedback": "  raw judge feedback\n"},
        reward=0.0,
        feedback_content="branch-local feedback",
    )

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    rendered = renderer.calls[0]["messages"][0]["content"]
    assert "  raw judge feedback\n" in rendered
    assert "branch-local feedback" not in rendered
    assert "user:   raw judge feedback" not in rendered


def test_sdpo_student_support_scores_teacher_on_prepopulated_candidate_ids(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        assert model_name == "policy-model"
        assert token_ids == [101, 102, 3, 4]
        assert candidate_token_ids == [[], [], [300, 301], [400, 401]]
        return [[], [], [-3.0, -13.0], [-4.0, -14.0]]

    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("student-support SDPO must not request teacher-selected top-k rows")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_topk_token_ids == [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]
    assert rollout.samples[0].sdpo_topk_logprobs == [
        [0.0, 0.0],
        [0.0, 0.0],
        [-3.0, -13.0],
        [-4.0, -14.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_sdpo_student_support_rejects_empty_teacher_prefix_before_prefill(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("empty SDPO teacher prefixes must fail before student-support candidate scoring")

    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("student-support SDPO must not request teacher-selected top-k rows")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="teacher reprompt rendered to an empty token prefix"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_scores_only_nonzero_sdpo_target_positions(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        assert token_ids == [101, 102, 3, 4]
        assert candidate_token_ids == [[], [], [], [400, 401]]
        return [[], [], [], [-4.0, -14.0]]

    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("student-support SDPO must not request teacher-selected top-k rows")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    sample = rollout.samples[0]
    sample.sdpo_weights = [0.0, 0.0, 0.0, 0.75, 0.0, 0.0]
    sample.sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert sample.sdpo_topk_token_ids == [[0, 0], [0, 0], [0, 0], [400, 401], [0, 0], [0, 0]]
    assert sample.sdpo_topk_logprobs == [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [-4.0, -14.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_sdpo_student_support_scores_teacher_on_hydrated_export_candidate_ids(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        assert model_name == "policy-model"
        assert token_ids == [101, 102, 3, 4]
        assert candidate_token_ids == [[], [], [300, 301], [400, 401]]
        return [[], [], [-3.0, -13.0], [-4.0, -14.0]]

    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("hydrated student-support SDPO must not request teacher-selected top-k rows")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    sample = rollout.samples[0]
    stamp_loss_routing(sample, "sdpo")
    hydrated = hydrate_student_support_from_records(
        [sample],
        [
            SDPOStudentSupportRecord(
                env_name="test-env",
                token_ids=[1, 2, 3, 4, 5, 6],
                position_ids=[0, 1, 2, 3, 4, 5],
                loss_mask=[False, False, True, True, False, False],
                sdpo_weights=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                student_topk_token_ids=[[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]],
                student_topk_logprobs=[[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4], [0.0, 0.0], [0.0, 0.0]],
            )
        ],
    )

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert hydrated == 2
    assert sample.sdpo_topk_token_ids == [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]
    assert sample.sdpo_topk_logprobs == [
        [0.0, 0.0],
        [0.0, 0.0],
        [-3.0, -13.0],
        [-4.0, -14.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_sdpo_student_support_chunks_candidate_scoring_when_union_exceeds_vllm_limit(monkeypatch):
    calls = []

    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        calls.append((token_ids, candidate_token_ids))
        assert model_name == "policy-model"
        if len(calls) == 1:
            assert token_ids == [101, 102, 3]
            assert candidate_token_ids == [[], [], [300, 301]]
            return [[], [], [-3.0, -13.0]]
        assert token_ids == [101, 102, 3, 4]
        assert candidate_token_ids == [[], [], [], [400, 401]]
        return [[], [], [], [-4.0, -14.0]]

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.MAX_PREFILL_CANDIDATE_TOKEN_IDS", 3)
    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="The attempt ignored the visible state.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert len(calls) == 2
    assert rollout.samples[0].sdpo_topk_logprobs == [
        [0.0, 0.0],
        [0.0, 0.0],
        [-3.0, -13.0],
        [-4.0, -14.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_sdpo_student_support_rejects_single_row_over_candidate_limit(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("oversized student row must fail before teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.MAX_PREFILL_CANDIDATE_TOKEN_IDS", 1)
    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="candidate-token limit"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_requires_prepopulated_candidate_ids(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("missing student support must fail before teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="hydrated student-selected support from the preflight token export"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


@pytest.mark.parametrize(
    "support_rows",
    [
        [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0]],
        [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0], [0, 0]],
    ],
)
def test_sdpo_student_support_rejects_candidate_stream_length_mismatch(monkeypatch, support_rows):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("misaligned student support must fail before teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = support_rows

    with pytest.raises(ValueError, match="student top-k stream length"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_candidate_row_width_mismatch(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("malformed student support must fail before teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="student top-k row width mismatch"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_non_list_candidate_row(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("malformed student support must fail before teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], (300, 301), [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="student top-k row must be a list"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_noninteger_candidate_token_ids(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("malformed student support must fail before teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, True], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="student top-k row contains non-integer token ids"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_negative_candidate_token_ids(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("malformed student support must fail before teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, -301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="student top-k row contains negative token ids"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_duplicate_candidate_token_ids(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("duplicate student support must fail before teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 300], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="student top-k row contains duplicate token ids"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_accepts_token_id_zero_candidate(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        assert candidate_token_ids == [[], [], [0], [400]]
        return [[], [], [-3.0], [-4.0]]

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=1, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0], [0], [0], [400], [0], [0]]

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_topk_token_ids == [[0], [0], [0], [400], [0], [0]]
    assert rollout.samples[0].sdpo_topk_logprobs == [[0.0], [0.0], [-3.0], [-4.0], [0.0], [0.0]]


def test_sdpo_student_support_rejects_teacher_candidate_response_row_count_mismatch(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        assert token_ids == [101, 102, 3, 4]
        return [[], [], [-3.0, -13.0]]

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="teacher response row count mismatch"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_teacher_candidate_response_row_width_mismatch(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        assert token_ids == [101, 102, 3, 4]
        return [[], [], [-3.0], [-4.0, -14.0]]

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="teacher response row width mismatch"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_non_list_teacher_candidate_response_row(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        assert token_ids == [101, 102, 3, 4]
        return [[], [], (-3.0, -13.0), [-4.0, -14.0]]

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="teacher response row must be a list"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_nonfinite_teacher_candidate_logprobs(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        return [[], [], [-3.0, float("nan")], [-4.0, -14.0]]

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="floating-point logprobs"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_boolean_teacher_candidate_logprobs(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        return [[], [], [-3.0, True], [-4.0, -14.0]]

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="floating-point logprobs"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_integer_teacher_candidate_logprobs(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        return [[], [], [-3, -13], [-4.0, -14.0]]

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="floating-point logprobs"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_student_support_rejects_teacher_candidate_logprob_mass_above_one(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        return [[], [], [0.0, -0.1], [-4.0, -14.0]]

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")
    rollout.samples[0].sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="probability mass exceeds 1"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_teacher_topk_row_count_mismatch(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return ([[30, 31]], [[-0.5, -1.5]])

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="teacher top-k row count mismatch"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_teacher_topk_row_width_mismatch(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="teacher top-k row width mismatch"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


@pytest.mark.parametrize(
    ("token_rows", "logprob_rows", "message"),
    [
        (
            [[0, 0], [0, 0], (30, 31), [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
            "teacher top-k token row must be a list",
        ),
        (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], (-0.5, -1.5), [-0.4, -1.4]],
            "teacher top-k logprob row must be a list",
        ),
    ],
)
def test_sdpo_rejects_teacher_topk_non_list_rows(monkeypatch, token_rows, logprob_rows, message):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (token_rows, logprob_rows)

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match=message):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_teacher_topk_noninteger_token_ids(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, True], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="teacher top-k row contains non-integer token ids"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_teacher_topk_negative_token_ids(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, -31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="teacher top-k row contains negative token ids"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_teacher_topk_duplicate_token_ids(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 30], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="teacher top-k row contains duplicate token ids"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_accepts_teacher_topk_token_id_zero(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0], [0], [0], [40]],
            [[0.0], [0.0], [-0.5], [-1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=1, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_topk_token_ids == [[0], [0], [0], [40], [0], [0]]
    assert rollout.samples[0].sdpo_topk_logprobs == [[0.0], [0.0], [-0.5], [-1.4], [0.0], [0.0]]


def test_sdpo_rejects_teacher_topk_placeholder_logprobs(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0], [0], [30], [40]],
            [[0.0], [0.0], [0.0], [-1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=1, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="teacher top-k row contains placeholder logprobs"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_teacher_topk_nonfinite_logprobs(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, float("inf")], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="floating-point logprobs"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_teacher_topk_boolean_logprobs(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, True], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="floating-point logprobs"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_teacher_topk_integer_logprobs(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-2, -3], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="floating-point logprobs"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_rejects_teacher_topk_logprob_mass_above_one(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, -0.1], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use feedback.")

    with pytest.raises(ValueError, match="probability mass exceeds 1"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))


def test_sdpo_prefers_renderer_assistant_prefill_when_available(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        assert token_ids == [501, 502, 3, 4]
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _AssistantPrefillRenderer(prefill_ids=[501, 502])
    algo = SDPOAlgorithm(
        _build(
            type="sdpo",
            distillation_topk=2,
            distillation_topk_support="teacher",
            template="{question}{feedback_block}",
            assistant_prefix="Correctly solve the original question.\n",
        ),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use the feedback.")

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert renderer.calls == []
    assert renderer.text_calls == []
    assert renderer.prefill_calls == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Solve the task.\n"
                        "The following is feedback from your unsuccessful earlier attempt:\n\n"
                        "user: Use the feedback."
                    ),
                }
            ],
            "text": "Correctly solve the original question.\n",
        }
    ]


def test_sdpo_renders_assistant_prefix_as_assistant_message_when_supported():
    renderer = _RenderAssistantPrefillRenderer()
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), renderer)

    prefix_ids = algo._assistant_prefill_ids([{"role": "user", "content": "Solve."}], "Use the hint.\n")

    assert prefix_ids == [101, 102, 201, 202]
    assert renderer.render_calls == [
        {
            "messages": [
                {"role": "user", "content": "Solve."},
                {"role": "assistant", "content": "Use the hint.\n"},
            ],
            "add_generation_prompt": False,
        }
    ]
    assert renderer.calls == []
    assert renderer.text_calls == []


def test_sdpo_rejects_malformed_renderer_assistant_prefill_attribution():
    renderer = _MalformedAssistantPrefillRenderer(sampled_mask=[False, True])
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), renderer)

    with pytest.raises(ValueError, match="renderer attribution must align"):
        algo._assistant_prefill_ids([{"role": "user", "content": "Solve."}], "Use the hint.\n")

    assert renderer.calls == []
    assert renderer.text_calls == []


def test_sdpo_rejects_renderer_assistant_prefill_without_close_token():
    renderer = _MalformedAssistantPrefillRenderer(close_token=998)
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), renderer)

    with pytest.raises(ValueError, match="assistant-prefill close token"):
        algo._assistant_prefill_ids([{"role": "user", "content": "Solve."}], "Use the hint.\n")

    assert renderer.calls == []
    assert renderer.text_calls == []


def test_sdpo_left_truncates_teacher_reprompt_prefix():
    renderer = _CaptureRenderer(token_ids=[101, 102, 103, 104, 105, 106], text_ids=[901, 902])
    algo = SDPOAlgorithm(_build(type="sdpo", max_reprompt_len=4, reprompt_truncation="left"), MagicMock(), renderer)

    prefix_ids = algo._teacher_prefix_ids(
        [{"role": "user", "content": "Solve the task."}],
        successful_previous_rollout="Correct attempt",
        hindsight_feedback="",
        env_name="test-env",
    )

    assert prefix_ids == [103, 104, 105, 106]


def test_sdpo_right_truncates_teacher_reprompt_prefix():
    renderer = _CaptureRenderer(token_ids=[101, 102, 103, 104, 105, 106], text_ids=[901, 902])
    algo = SDPOAlgorithm(_build(type="sdpo", max_reprompt_len=4, reprompt_truncation="right"), MagicMock(), renderer)

    prefix_ids = algo._teacher_prefix_ids(
        [{"role": "user", "content": "Solve the task."}],
        successful_previous_rollout="Correct attempt",
        hindsight_feedback="",
        env_name="test-env",
    )

    assert prefix_ids == [101, 102, 103, 104]


def test_sdpo_error_truncation_rejects_long_teacher_reprompt():
    renderer = _CaptureRenderer(token_ids=[101, 102, 103, 104, 105, 106], text_ids=[901, 902])
    algo = SDPOAlgorithm(_build(type="sdpo", max_reprompt_len=4, reprompt_truncation="error"), MagicMock(), renderer)

    with pytest.raises(ValueError, match="teacher reprompt exceeds max_reprompt_len=4"):
        algo._teacher_prefix_ids(
            [{"role": "user", "content": "Solve the task."}],
            successful_previous_rollout="Correct attempt",
            hindsight_feedback="",
            env_name="test-env",
        )


def test_sdpo_default_reprompt_matches_huebotter_yaml_template_spacing():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), renderer)

    algo._teacher_prefix_ids(
        [{"role": "user", "content": "Solve the task."}],
        successful_previous_rollout="Correct attempt",
        hindsight_feedback="tool: The attempt missed a case.",
        env_name="test-env",
    )

    assert renderer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": (
                "Solve the task.\n"
                "Correct solution:\n\n"
                "Correct attempt\n"
                "The following is feedback from your unsuccessful earlier attempt:\n\n"
                "tool: The attempt missed a case.\n\n"
                "Correctly solve the original question."
            ),
        }
    ]


def test_sdpo_default_reprompt_rewrites_first_user_to_preserve_feedback_messages():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), renderer)

    algo._teacher_prefix_ids(
        [
            {"role": "system", "content": "System instruction."},
            {"role": "user", "content": "Original task."},
            {"role": "assistant", "content": "Failed attempt."},
            {"role": "user", "content": "Runtime feedback to preserve."},
        ],
        successful_previous_rollout="Correct attempt",
        hindsight_feedback="",
        env_name="test-env",
    )

    assert renderer.calls[0]["messages"] == [
        {"role": "system", "content": "System instruction."},
        {
            "role": "user",
            "content": "Original task.\nCorrect solution:\n\nCorrect attempt\n\nCorrectly solve the original question.",
        },
        {"role": "assistant", "content": "Failed attempt."},
        {"role": "user", "content": "Runtime feedback to preserve."},
    ]


def test_sdpo_can_rewrite_last_user_prompt_when_configured():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(_build(type="sdpo", template_target="last_user"), MagicMock(), renderer)

    algo._teacher_prefix_ids(
        [
            {"role": "system", "content": "System instruction."},
            {"role": "user", "content": "Earlier user context."},
            {"role": "assistant", "content": "Clarifying turn."},
            {"role": "user", "content": "Latest task prompt."},
        ],
        successful_previous_rollout="Correct attempt",
        hindsight_feedback="",
        env_name="test-env",
    )

    assert renderer.calls[0]["messages"] == [
        {"role": "system", "content": "System instruction."},
        {"role": "user", "content": "Earlier user context."},
        {"role": "assistant", "content": "Clarifying turn."},
        {
            "role": "user",
            "content": "Latest task prompt.\nCorrect solution:\n\nCorrect attempt\n\nCorrectly solve the original question.",
        },
    ]


def test_sdpo_custom_section_templates_flow_into_reprompt():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(
            type="sdpo",
            solution_template="\nDemonstration:\n{successful_previous_attempt}",
            feedback_template="\nVerifier feedback:\n{feedback_raw}",
        ),
        MagicMock(),
        renderer,
    )

    algo._teacher_prefix_ids(
        [{"role": "user", "content": "Solve the task."}],
        successful_previous_rollout="Correct attempt",
        hindsight_feedback="tool: The attempt missed a case.",
        env_name="test-env",
    )

    assert renderer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": (
                "Solve the task.\n"
                "Demonstration:\n"
                "Correct attempt\n"
                "Verifier feedback:\n"
                "tool: The attempt missed a case.\n\n"
                "Correctly solve the original question."
            ),
        }
    ]


def test_sdpo_skips_failed_sample_without_solution_or_feedback(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("no-target SDPO samples must not request teacher top-k scores")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0)

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_weights == [0.0, 0.0, 0.0, 0.0]
    assert rollout.samples[0].sdpo_topk_token_ids is None
    assert rollout.samples[0].sdpo_topk_logprobs is None
    assert renderer.calls == []
    assert renderer.text_calls == []


def test_sdpo_no_target_clears_prepopulated_student_support(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("no-target SDPO samples must not request teacher candidate scores")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0)
    sample = rollout.samples[0]
    sample.sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [0.0, 0.0], [-3.0, -13.0], [-4.0, -14.0]]
    sample.sdpo_rollout_is_weights = [1.0, 1.0, 0.7, 0.8]

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert sample.sdpo_weights == [0.0, 0.0, 0.0, 0.0]
    assert sample.sdpo_topk_token_ids is None
    assert sample.sdpo_topk_logprobs is None
    assert sample.sdpo_rollout_is_weights is None
    assert renderer.calls == []


def test_sdpo_student_support_preflight_clears_no_hindsight_target():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout(reward=0.0)
    sample = rollout.samples[0]
    sample.sdpo_weights = [0.0, 0.0, 1.0, 1.0]
    sample.sdpo_topk_token_ids = [[0, 0], [0, 0], [300, 301], [400, 401]]
    sample.sdpo_topk_logprobs = [[0.0, 0.0], [0.0, 0.0], [-3.0, -13.0], [-4.0, -14.0]]
    sample.sdpo_rollout_is_weights = [1.0, 1.0, 0.7, 0.8]

    selected = algo.select_batch_preflight_samples(
        [RolloutView(rollout)],
        samples=[sample],
    )

    assert selected == []
    assert sample.sdpo_weights == [0.0, 0.0, 0.0, 0.0]
    assert sample.sdpo_topk_token_ids is None
    assert sample.sdpo_topk_logprobs is None
    assert sample.sdpo_rollout_is_weights is None


def test_sdpo_student_support_preflight_skips_self_success_with_feedback(tmp_path, monkeypatch):
    async def fake_preflight(**kwargs):
        raise AssertionError("self-success feedback must not trigger student-support preflight")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.run_sdpo_student_support_preflight", fake_preflight)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout(
        reward=1.0,
        assistant_content="Correct attempt",
        feedback_content="The grader supplied extra feedback.",
    )
    sample = rollout.samples[0]
    sample.sdpo_weights = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

    selected = algo.select_batch_preflight_samples(
        [RolloutView(rollout)],
        samples=[sample],
    )

    assert selected == []
    assert sample.sdpo_weights == [0.0] * len(sample.token_ids)


def test_sdpo_student_support_preflight_respects_provided_sample_scope():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use the feedback.")
    rollout.samples[0].sdpo_weights = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    unrelated_sample = SimpleNamespace(sdpo_weights=[1.0])

    selected = algo.select_batch_preflight_samples(
        [RolloutView(rollout)],
        samples=[unrelated_sample],
    )

    assert selected == []
    assert rollout.samples[0].sdpo_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_sdpo_student_support_preflight_does_not_clear_self_success_outside_sample_scope():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout(
        reward=1.0,
        assistant_content="Correct attempt",
        feedback_content="The grader supplied extra feedback.",
    )
    sample = rollout.samples[0]
    sample.sdpo_weights = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    unrelated_sample = SimpleNamespace(sdpo_weights=[1.0])

    selected = algo.select_batch_preflight_samples(
        [RolloutView(rollout)],
        samples=[unrelated_sample],
    )

    assert selected == []
    assert sample.sdpo_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_sdpo_student_support_preflight_does_not_default_weights_outside_sample_scope():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use the feedback.")
    sample = rollout.samples[0]
    sample.sdpo_weights = None
    unrelated_sample = SimpleNamespace(sdpo_weights=[1.0])

    selected = algo.select_batch_preflight_samples(
        [RolloutView(rollout)],
        samples=[unrelated_sample],
    )

    assert selected == []
    assert sample.sdpo_weights is None


def test_sdpo_student_support_preflight_defaults_missing_sdpo_weights_to_sampled_tokens():
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student"),
        MagicMock(),
        renderer,
    )
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use the feedback.")
    sample = rollout.samples[0]
    sample.sdpo_weights = None

    selected = algo.select_batch_preflight_samples(
        [RolloutView(rollout)],
        samples=[sample],
    )

    assert selected == [sample]
    assert sample.sdpo_weights == [1.0 if trains else 0.0 for trains in sample.mask]


def test_sdpo_student_support_preflight_zeroes_multi_turn_no_target_segments(tmp_path, monkeypatch):
    received = {}

    async def fake_preflight(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.run_sdpo_student_support_preflight", fake_preflight)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student", multi_turn=True),
        MagicMock(),
        renderer,
    )
    rollout = _two_turn_rollout(reward=0.0)
    sample = rollout.samples[0]
    sample.sdpo_weights = [1.0 if trains else 0.0 for trains in sample.mask]
    sample.sdpo_rollout_is_weights = [0.0, 0.0, 0.7, 0.8, 0.0, 0.0, 0.9, 1.1]
    sample.sdpo_topk_token_ids = [
        [0, 0],
        [0, 0],
        [300, 301],
        [400, 401],
        [0, 0],
        [0, 0],
        [700, 701],
        [800, 801],
    ]
    sample.sdpo_topk_logprobs = [
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.3, -1.3],
        [-0.4, -1.4],
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.7, -1.7],
        [-0.8, -1.8],
    ]

    selected = algo.select_batch_preflight_samples(
        [RolloutView(rollout)],
        samples=[sample],
    )

    asyncio.run(
        algo.run_batch_preflight(
            [RolloutView(rollout)],
            samples=selected,
            output_dir=tmp_path,
            sender=MagicMock(),
            step=7,
        )
    )

    assert received["samples"] == [sample]
    assert received["expected_topk"] == 2
    assert sample.sdpo_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert sample.sdpo_rollout_is_weights == [0.0, 0.0, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0]
    assert sample.sdpo_topk_token_ids == [
        [0, 0],
        [0, 0],
        [300, 301],
        [400, 401],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert sample.sdpo_topk_logprobs == [
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.3, -1.3],
        [-0.4, -1.4],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_sdpo_multi_turn_student_support_scores_hydrated_preflight_targets(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        assert model_name == "policy-model"
        assert token_ids == [101, 102, 3, 4]
        assert candidate_token_ids == [[], [], [300, 301], [400, 401]]
        return [[], [], [-3.0, -13.0], [-4.0, -14.0]]

    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("multi-turn student-support SDPO must not request teacher-selected top-k rows")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202], [301, 302]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student", multi_turn=True),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout(reward=0.0)
    sample = rollout.samples[0]
    sample.sdpo_weights = [1.0 if trains else 0.0 for trains in sample.mask]

    selected = algo.select_batch_preflight_samples([RolloutView(rollout)], samples=[sample])

    assert selected == [sample]
    assert sample.sdpo_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    hydrated = hydrate_student_support_from_records(
        [sample],
        [
            SDPOStudentSupportRecord(
                env_name="test-env",
                token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                position_ids=[0, 1, 2, 3, 4, 5, 6, 7],
                loss_mask=[False, False, True, True, False, False, True, True],
                sdpo_weights=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                student_topk_token_ids=[
                    [0, 0],
                    [0, 0],
                    [300, 301],
                    [400, 401],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                student_topk_logprobs=[
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [-0.5, -1.5],
                    [-0.4, -1.4],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            )
        ],
        expected_topk=2,
    )

    sample.sdpo_topk_token_ids[6] = [700, 701]
    sample.sdpo_topk_token_ids[7] = [800, 801]

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert hydrated == 2
    assert sample.sdpo_topk_token_ids == [
        [0, 0],
        [0, 0],
        [300, 301],
        [400, 401],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert sample.sdpo_topk_logprobs == [
        [0.0, 0.0],
        [0.0, 0.0],
        [-3.0, -13.0],
        [-4.0, -14.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    assert len(renderer.calls) == 1


def test_sdpo_multi_turn_rewrites_original_prompt_without_overwriting_user_feedback(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        assert model_name == "policy-model"
        assert topk == 2
        if token_ids == [101, 102, 3, 4]:
            return (
                [[0, 0], [0, 0], [30, 31], [40, 41]],
                [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
            )
        assert token_ids == [201, 202, 7, 8]
        return (
            [[0, 0], [0, 0], [70, 71], [80, 81]],
            [[0.0, 0.0], [0.0, 0.0], [-0.7, -1.7], [-0.8, -1.8]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher", multi_turn=True),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    group_id = uuid.uuid4()
    failed = _two_turn_rollout(observation_role="user", reward=0.0)
    failed.group_id = group_id
    successful = _single_turn_rollout(reward=1.0, assistant_content="Correct attempt", group_id=group_id)

    asyncio.run(algo.score_batch([RolloutView(failed), RolloutView(successful)]))

    assert renderer.calls[1]["messages"] == [
        {
            "role": "user",
            "content": "U\nCorrect solution:\n\nCorrect attempt\n\nCorrectly solve the original question.",
        },
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "feedback"},
    ]


def test_sdpo_uses_successful_sibling_as_correct_solution(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    group_id = uuid.uuid4()
    failed = _single_turn_rollout(reward=0.0, assistant_content="Wrong attempt", group_id=group_id)
    successful = _single_turn_rollout(reward=1.0, assistant_content="Correct attempt", group_id=group_id)

    asyncio.run(algo.score_batch([RolloutView(failed), RolloutView(successful)]))

    assert renderer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": "Solve the task.\nCorrect solution:\n\nCorrect attempt\n\nCorrectly solve the original question.",
        }
    ]
    assert renderer.text_calls == []
    assert "unsuccessful earlier attempt" not in renderer.calls[0]["messages"][0]["content"]


def test_sdpo_uses_first_successful_sibling_by_default(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202], [301, 302]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    group_id = uuid.uuid4()
    failed = _single_turn_rollout(reward=0.0, assistant_content="Wrong attempt", group_id=group_id)
    weaker = _single_turn_rollout(reward=0.6, assistant_content="Partly correct attempt", group_id=group_id)
    stronger = _single_turn_rollout(reward=1.0, assistant_content="Best correct attempt", group_id=group_id)

    asyncio.run(algo.score_batch([RolloutView(failed), RolloutView(weaker), RolloutView(stronger)]))

    assert renderer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": "Solve the task.\nCorrect solution:\n\nPartly correct attempt\n\nCorrectly solve the original question.",
        }
    ]


def test_sdpo_can_prefer_highest_reward_successful_sibling_when_configured(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202], [301, 302]])
    algo = SDPOAlgorithm(
        _build(
            type="sdpo",
            distillation_topk=2,
            distillation_topk_support="teacher",
            successful_demonstration_selection="highest_reward",
        ),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    group_id = uuid.uuid4()
    failed = _single_turn_rollout(reward=0.0, assistant_content="Wrong attempt", group_id=group_id)
    weaker = _single_turn_rollout(reward=0.6, assistant_content="Partly correct attempt", group_id=group_id)
    stronger = _single_turn_rollout(reward=1.0, assistant_content="Best correct attempt", group_id=group_id)

    asyncio.run(algo.score_batch([RolloutView(failed), RolloutView(weaker), RolloutView(stronger)]))

    assert renderer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": "Solve the task.\nCorrect solution:\n\nBest correct attempt\n\nCorrectly solve the original question.",
        }
    ]


def test_sdpo_does_not_use_success_from_different_group_with_same_task_idx(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("different Prime groups must not share SDPO solutions")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    failed = _single_turn_rollout(reward=0.0, assistant_content="Wrong attempt", group_id=uuid.uuid4())
    successful = _single_turn_rollout(reward=1.0, assistant_content="Correct attempt", group_id=uuid.uuid4())

    asyncio.run(algo.score_batch([RolloutView(failed), RolloutView(successful)]))

    assert failed.samples[0].sdpo_weights == [0.0, 0.0, 0.0, 0.0]
    assert renderer.calls == []


def test_sdpo_empty_successful_sibling_counts_as_solution_and_suppresses_feedback(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    group_id = uuid.uuid4()
    failed = _single_turn_rollout(
        reward=0.0,
        assistant_content="Wrong attempt",
        feedback_content="Do not include this feedback when a solution exists.",
        group_id=group_id,
    )
    successful = _single_turn_rollout(reward=1.0, assistant_content="", group_id=group_id)

    asyncio.run(algo.score_batch([RolloutView(failed), RolloutView(successful)]))

    rendered = renderer.calls[0]["messages"][0]["content"]
    assert rendered == "Solve the task.\nCorrect solution:\n\n\n\nCorrectly solve the original question."
    assert "Do not include this feedback when a solution exists." not in rendered


def test_sdpo_thinking_only_successful_sibling_counts_as_empty_solution_after_stripping(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    group_id = uuid.uuid4()
    failed = _single_turn_rollout(
        reward=0.0,
        assistant_content="Wrong attempt",
        feedback_content="Use the visible feedback.",
        group_id=group_id,
    )
    successful = _single_turn_rollout(
        reward=1.0,
        assistant_content="<think>hidden scratchpad</think>\n",
        group_id=group_id,
    )

    asyncio.run(algo.score_batch([RolloutView(failed), RolloutView(successful)]))

    rendered = renderer.calls[0]["messages"][0]["content"]
    assert rendered == "Solve the task.\nCorrect solution:\n\n\n\nCorrectly solve the original question."
    assert "Use the visible feedback." not in rendered


def test_sdpo_single_turn_prefers_structured_feedback_over_branch_fallback(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(
        info={"feedback": "Structured verifier feedback."},
        reward=0.0,
        feedback_content="Branch-local fallback feedback.",
    )

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    rendered = renderer.calls[0]["messages"][0]["content"]
    assert "Structured verifier feedback." in rendered
    assert "Branch-local fallback feedback." not in rendered


def test_sdpo_success_threshold_matches_huebotter_sdpo_config():
    partial = RolloutView(_single_turn_rollout(reward=0.5, assistant_content="Partial attempt"))

    default_algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), _CaptureRenderer(token_ids=[101, 102]))
    assert default_algo._successful_previous_rollouts([partial]) == {
        partial.raw.group_id: [(partial, "Partial attempt")]
    }

    strict_algo = SDPOAlgorithm(
        _build(type="sdpo", success_reward_threshold=1.0), MagicMock(), _CaptureRenderer(token_ids=[101, 102])
    )
    assert strict_algo._successful_previous_rollouts([partial]) == {}


def test_sdpo_skips_self_success_by_default(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("self-success without another solution should not be reprompted by default")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=1.0, assistant_content="Correct attempt")

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_weights == [0.0, 0.0, 0.0, 0.0]
    assert renderer.calls == []


def test_sdpo_skips_self_success_even_with_feedback_by_default(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("self-success feedback must not create an SDPO target by default")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher"), MagicMock(), renderer
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(
        reward=1.0,
        assistant_content="Correct attempt",
        feedback_content="The grader supplied extra feedback.",
    )

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_weights == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert rollout.samples[0].sdpo_topk_token_ids is None
    assert rollout.samples[0].sdpo_topk_logprobs is None
    assert renderer.calls == []


def test_sdpo_can_reprompt_on_self_success_when_enabled(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        assert token_ids == [101, 102, 3, 4]
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(
            type="sdpo",
            distillation_topk=2,
            distillation_topk_support="teacher",
            dont_reprompt_on_self_success=False,
        ),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=1.0, assistant_content="Correct attempt")

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert (
        renderer.calls[0]["messages"][0]["content"]
        == "Solve the task.\nCorrect solution:\n\nCorrect attempt\n\nCorrectly solve the original question."
    )


def test_sdpo_can_disable_environment_feedback(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("feedback-disabled failed samples should not request teacher scores")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(
            type="sdpo",
            distillation_topk=2,
            distillation_topk_support="teacher",
            include_environment_feedback=False,
        ),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(reward=0.0, feedback_content="Use the feedback.")

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_weights == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert renderer.calls == []


def test_sdpo_can_include_feedback_even_with_solution(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[101, 102])
    algo = SDPOAlgorithm(
        _build(
            type="sdpo",
            distillation_topk=2,
            distillation_topk_support="teacher",
            dont_reprompt_on_self_success=False,
            environment_feedback_only_without_solution=False,
        ),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _single_turn_rollout(
        reward=1.0,
        assistant_content="Correct attempt",
        feedback_content="The grader supplied extra feedback.",
    )

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    rendered = renderer.calls[0]["messages"][0]["content"]
    assert "Correct solution:\n\nCorrect attempt" in rendered
    assert "user: The grader supplied extra feedback." in rendered


def test_sdpo_removes_thinking_trace_from_demonstrations():
    rollout = RolloutView(
        _single_turn_rollout(reward=1.0, assistant_content="<think>hidden scratchpad</think>\nFinal answer")
    )
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), _CaptureRenderer(token_ids=[101, 102]))

    assert algo._successful_previous_rollouts([rollout]) == {rollout.raw.group_id: [(rollout, "Final answer")]}


def test_sdpo_can_preserve_thinking_trace_in_demonstrations_when_configured():
    rollout = RolloutView(
        _single_turn_rollout(reward=1.0, assistant_content="<think>hidden scratchpad</think>\nFinal answer")
    )
    algo = SDPOAlgorithm(
        _build(type="sdpo", remove_thinking_from_demonstration=False),
        MagicMock(),
        _CaptureRenderer(token_ids=[101, 102]),
    )

    assert algo._successful_previous_rollouts([rollout]) == {
        rollout.raw.group_id: [(rollout, "<think>hidden scratchpad</think>\nFinal answer")]
    }


def test_sdpo_preserves_thinking_only_successful_demonstration_as_empty_solution_after_stripping():
    rollout = RolloutView(_single_turn_rollout(reward=1.0, assistant_content="<think>hidden scratchpad</think>\n"))
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), _CaptureRenderer(token_ids=[101, 102]))

    assert algo._successful_previous_rollouts([rollout]) == {rollout.raw.group_id: [(rollout, "")]}


def test_sdpo_preserves_successful_demonstration_whitespace():
    rollout = RolloutView(_single_turn_rollout(reward=1.0, assistant_content="  indented answer\n"))
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), _CaptureRenderer(token_ids=[101, 102]))

    assert algo._successful_previous_rollouts([rollout]) == {rollout.raw.group_id: [(rollout, "  indented answer\n")]}


def test_sdpo_preserves_whitespace_only_successful_demonstration_text():
    rollout = RolloutView(_single_turn_rollout(reward=1.0, assistant_content=" \n "))
    algo = SDPOAlgorithm(_build(type="sdpo"), MagicMock(), _CaptureRenderer(token_ids=[101, 102]))

    assert algo._successful_previous_rollouts([rollout]) == {rollout.raw.group_id: [(rollout, " \n ")]}


def test_sdpo_multi_turn_uses_branch_local_feedback_when_no_solution(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        assert token_ids == [101, 102, 3, 4]
        return (
            [[0, 0], [0, 0], [30, 31], [40, 41]],
            [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher", multi_turn=True),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout(reward=0.0)

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_topk_token_ids == [
        [0, 0],
        [0, 0],
        [30, 31],
        [40, 41],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert rollout.samples[0].sdpo_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert "tool: T" in renderer.calls[0]["messages"][0]["content"]
    assert "unsuccessful earlier attempt" in renderer.calls[0]["messages"][0]["content"]
    assert len(renderer.calls) == 1


def test_sdpo_multi_turn_teacher_support_rejects_precomputed_rollout_is_weights(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("stale rollout-IS weights must fail before multi-turn teacher scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher", multi_turn=True),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout(reward=0.0)
    sample = rollout.samples[0]
    original_topk_ids = [
        [0, 0],
        [0, 0],
        [300, 301],
        [400, 401],
        [0, 0],
        [0, 0],
        [700, 701],
        [800, 801],
    ]
    original_topk_logprobs = [
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.3, -1.3],
        [-0.4, -1.4],
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.7, -1.7],
        [-0.8, -1.8],
    ]
    sample.sdpo_topk_token_ids = [list(row) for row in original_topk_ids]
    sample.sdpo_topk_logprobs = [list(row) for row in original_topk_logprobs]
    sample.sdpo_rollout_is_weights = [0.0, 0.0, 0.7, 0.8, 0.0, 0.0, 0.9, 1.1]

    with pytest.raises(ValueError, match="rollout-IS weights"):
        asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert sample.sdpo_topk_token_ids == original_topk_ids
    assert sample.sdpo_topk_logprobs == original_topk_logprobs


def test_sdpo_multi_turn_uses_successful_sibling_when_branch_feedback_is_absent(monkeypatch):
    calls = []

    async def fake_prefill_topk(client, model_name, token_ids, topk):
        calls.append(token_ids)
        assert model_name == "policy-model"
        if len(calls) == 1:
            assert token_ids == [101, 102, 3, 4]
            return (
                [[0, 0], [0, 0], [30, 31], [40, 41]],
                [[0.0, 0.0], [0.0, 0.0], [-0.5, -1.5], [-0.4, -1.4]],
            )
        assert token_ids == [201, 202, 7, 8]
        return (
            [[0, 0], [0, 0], [70, 71], [80, 81]],
            [[0.0, 0.0], [0.0, 0.0], [-0.7, -1.7], [-0.8, -1.8]],
        )

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher", multi_turn=True),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    group_id = uuid.uuid4()
    failed = _two_turn_rollout(reward=0.0)
    failed.group_id = group_id
    failed.nodes[2].message.content = ""
    successful = _two_turn_rollout(reward=1.0)
    successful.group_id = group_id

    asyncio.run(algo.score_batch([RolloutView(failed), RolloutView(successful)]))

    assert calls == [[101, 102, 3, 4], [201, 202, 7, 8]]
    assert failed.samples[0].sdpo_topk_token_ids == [
        [0, 0],
        [0, 0],
        [30, 31],
        [40, 41],
        [0, 0],
        [0, 0],
        [70, 71],
        [80, 81],
    ]
    assert failed.samples[0].sdpo_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert len(renderer.calls) == 2
    assert "Correct solution:\n\nA\nA2" in renderer.calls[0]["messages"][0]["content"]
    assert "unsuccessful earlier attempt" not in renderer.calls[0]["messages"][0]["content"]
    assert "Correct solution:\n\nA\nA2" in renderer.calls[1]["messages"][0]["content"]


def test_sdpo_multi_turn_student_support_clears_no_target_without_preflight_support(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("no-target SDPO sample must not request teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student", multi_turn=True),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout(reward=0.0)
    rollout.nodes[2].message.content = ""

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    sample = rollout.samples[0]
    assert sample.sdpo_weights == [0.0] * len(sample.token_ids)
    assert sample.sdpo_topk_token_ids is None
    assert sample.sdpo_topk_logprobs is None


def test_sdpo_multi_turn_student_support_clears_stale_no_target_payload(monkeypatch):
    async def fake_prefill_candidates(client, model_name, token_ids, candidate_token_ids):
        raise AssertionError("stale no-target SDPO support must not request teacher candidate scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_candidate_logprobs", fake_prefill_candidates)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="student", multi_turn=True),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout(reward=0.0)
    rollout.nodes[2].message.content = ""
    sample = rollout.samples[0]
    sample.sdpo_weights = [1.0 if trains else 0.0 for trains in sample.mask]
    sample.sdpo_rollout_is_weights = [0.0, 0.0, 0.7, 0.8, 0.0, 0.0, 0.9, 1.1]
    sample.sdpo_topk_token_ids = [
        [0, 0],
        [0, 0],
        [300, 301],
        [400, 401],
        [0, 0],
        [0, 0],
        [700, 701],
        [800, 801],
    ]
    sample.sdpo_topk_logprobs = [
        [0.0, 0.0],
        [0.0, 0.0],
        [-3.0, -13.0],
        [-4.0, -14.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [-7.0, -17.0],
        [-8.0, -18.0],
    ]

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert sample.sdpo_weights == [0.0] * len(sample.token_ids)
    assert sample.sdpo_rollout_is_weights is None
    assert sample.sdpo_topk_token_ids is None
    assert sample.sdpo_topk_logprobs is None


def test_sdpo_multi_turn_teacher_support_clears_no_target_payload(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("no-target SDPO sample must not request teacher top-k scoring")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher", multi_turn=True),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout(reward=0.0)
    rollout.nodes[2].message.content = ""

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    sample = rollout.samples[0]
    assert sample.sdpo_weights == [0.0] * len(sample.token_ids)
    assert sample.sdpo_topk_token_ids is None
    assert sample.sdpo_topk_logprobs is None


def test_sdpo_multi_turn_skips_self_success_even_with_branch_feedback(monkeypatch):
    async def fake_prefill_topk(client, model_name, token_ids, topk):
        raise AssertionError("self-success branch feedback must not create an SDPO target by default")

    monkeypatch.setattr("prime_rl.orchestrator.algo.sdpo.compute_prefill_topk_logprobs", fake_prefill_topk)
    renderer = _CaptureRenderer(token_ids=[[101, 102], [201, 202]])
    algo = SDPOAlgorithm(
        _build(type="sdpo", distillation_topk=2, distillation_topk_support="teacher", multi_turn=True),
        MagicMock(),
        renderer,
    )
    algo.teacher_pool = SimpleNamespace(model_name="policy-model", train_clients=[object()])
    rollout = _two_turn_rollout(reward=1.0)

    asyncio.run(algo.score_batch([RolloutView(rollout)]))

    assert rollout.samples[0].sdpo_weights == [0.0] * len(rollout.samples[0].token_ids)
    assert rollout.samples[0].sdpo_topk_token_ids is None
    assert rollout.samples[0].sdpo_topk_logprobs is None
    assert renderer.calls == []
