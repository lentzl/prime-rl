"""Algorithm abstraction: sampling and the per-token training signal.

An algorithm is a named, self-contained config — a discriminated union keyed
on ``type`` (``grpo``, ``max_rl``, ``opd``, ``opsd``, ``sdpo``, ``sft``,
``echo``, ``reward``, ``custom``). The bundle *is* the algorithm: each variant carries
its sampling component and its credit-assignment / loss-routing parameters,
and its class defaults are the vetted setting — ``type = "opd"`` with a
teacher IS on-policy distillation; any key you set is visibly your own
assembly. There is no separate ``advantage`` sub-component and no preset layer.

Each algorithm fixes two things:

1. **Sampling** — which model generates train rollouts. ``sampling.source`` is
   a model reference: ``"policy"`` (the live policy) or an inline frozen hosted
   model.
2. **The per-token training signal** — credit assignment and loss routing,
   fused: one mapping from a finalized rollout to per-token ``(loss component,
   weight)``. Group-relative algorithms compute scalars on the orchestrator and
   ship numbers; reference-KL algorithms ship reference prefill logprobs and the
   trainer evaluates the per-token signal against the live policy. The algorithm
   determines which loss component consumes the action tokens (``rl`` / ``ce`` /
   ``ref_kl`` / ``sdpo``, via the ``action_loss_type`` class declaration) and what happens
   to env-provided observation tokens (masked out by default; ``echo`` trains on
   them with weighted CE).

prime-rl's primary hosted model is the trainable policy. Other ordinary model
references are external OpenAI-compatible endpoints, declared inline on the
algorithm that uses them (a :class:`FrozenModelConfig`). Internal live-derived
views, such as an SDPO EMA teacher, need their own weight-update lifecycle and
should not be encoded as frozen references. Model roles like
"teacher" are algorithm-local vocabulary over these references; the trainer is
algorithm-blind: the loss is a sum of four components (rl, ce, ref_kl, sdpo),
each normalized by its own global token count; per-token component weights ship
on the wire and the trainer just executes them.
"""

from string import Formatter
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.config import BaseConfig


class FrozenModelConfig(ClientConfig):
    """An externally hosted model behind an OpenAI-compatible endpoint: the
    client config plus the served model's ``name``.

    prime-rl never launches or updates these. Frozen models are
    reachable-but-unmanaged: ``base_url`` is required, their weights never
    change, and rollouts or scores from them never go stale (stable prefix
    cache, no off-policy aging)."""

    name: str
    """Served model name, sent as the ``model`` field of every request."""

    @model_validator(mode="after")
    def require_explicit_endpoint(self):
        if "base_url" not in self.model_fields_set and not self.is_elastic:
            raise ValueError(
                "a frozen model reference needs base_url — frozen models are externally "
                "hosted and are not updated by prime-rl."
            )
        return self


ModelReference: TypeAlias = Literal["policy"] | FrozenModelConfig
"""``"policy"`` (the live policy — weight-updated: prefix caches salted per
version, sampling logprobs carried, rollouts age off-policy) or an inline
externally-hosted frozen model."""

ActionLossType: TypeAlias = Literal["rl", "ce", "ref_kl", "sdpo"]

SDPO_MAX_STUDENT_SUPPORT_TOPK = 128
"""Maximum per-token student-support candidates vLLM can score in one request."""


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class SamplingConfig(BaseConfig):
    source: ModelReference = "policy"
    """Model reference for train rollout generation: ``"policy"`` (the live
    policy — prefix caches salted per version, sampling logprobs requested,
    rollouts age off-policy) or an inline frozen hosted model (stable prefix
    cache, no sampling logprobs, rollouts never go stale)."""


# ---------------------------------------------------------------------------
# Shared sub-configs (length penalty, echo roles)
# ---------------------------------------------------------------------------


class TokensLengthPenaltyConfig(BaseConfig):
    type: Literal["tokens"] = "tokens"

    completion_weight: float = Field(1.0, ge=0, allow_inf_nan=False)
    """Weight on model completion tokens. Finite and non-negative."""

    tool_response_weight: float = Field(1.0, ge=0, allow_inf_nan=False)
    """Weight on tool-response tokens (read from the rollout's ``*_total_tool_response_tokens`` harness metric; 0 if absent). Finite and non-negative."""


class TurnsLengthPenaltyConfig(BaseConfig):
    type: Literal["turns"] = "turns"


LengthPenaltyConfig: TypeAlias = Annotated[
    TokensLengthPenaltyConfig | TurnsLengthPenaltyConfig,
    Field(discriminator="type"),
]


class EchoRoleConfig(BaseConfig):
    """Echo CE supervision for one message role."""

    alpha: float = Field(0.1, gt=0)
    """Per-token ce weight for this role's env-provided tokens (ECHO's lambda)."""


class EchoRolesConfig(BaseConfig):
    """Which env-provided message roles train, each at its own weight.
    Setting any role replaces the whole table — unset roles stay disabled."""

    system: EchoRoleConfig | None = None
    user: EchoRoleConfig | None = None
    assistant: EchoRoleConfig | None = None
    tool: EchoRoleConfig | None = None

    @model_validator(mode="after")
    def require_a_role(self):
        if self.system is None and self.user is None and self.assistant is None and self.tool is None:
            raise ValueError("echo needs at least one role enabled (system, user, assistant, or tool)")
        return self


class EchoFilterConfig(BaseConfig):
    """User-supplied per-token filter narrowing the role-selected echo tokens.

    The callable is imported at startup and invoked once per rollout as
    ``filter_fn(rollout, **kwargs) -> list[list[bool]]`` — one keep-mask per
    trajectory step, each spanning that step's ``prompt_ids`` +
    ``completion_ids``. Tokens with ``False`` never receive echo weight; the
    filter can only narrow the role selection, not widen it. The raw rollout
    exposes message text and sampling logprobs, so content filters (e.g.
    dropping tool-output warnings) and sampling-probability filters need no
    extra framework surface."""

    import_path: str
    """Import path to the filter callable (e.g. ``my_module.drop_warnings``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the filter."""


# ---------------------------------------------------------------------------
# The algorithms (a discriminated union keyed on ``type``)
# ---------------------------------------------------------------------------


class BaseAlgorithmConfig(BaseConfig):
    """Base for every algorithm: the shared sampling component, the ``teacher``
    shorthand, and the reference-folding / compatibility validators. Each
    subclass sets ``type`` (the discriminator), declares its loss routing
    (``action_loss_type``), names its reference's role if it has one
    (``model_role`` / ``source_role``), and adds its own parameters.

    The bundle IS the algorithm — there is no separate ``advantage``
    sub-component. ``algo.type`` names it, and the class defaults are the
    vetted setting."""

    action_loss_type: ClassVar[ActionLossType] = "rl"

    sampling: SamplingConfig = SamplingConfig()
    """Sampling component: which model generates train rollouts."""

    teacher: ModelReference | None = Field(None, exclude=True)
    """Reference-model shorthand: an inline frozen hosted model (``name`` +
    ``base_url``). Folds into the slot the algorithm declares for it —
    ``model`` for the distillation algorithms (opd/opsd), ``sampling.source``
    for sft. ``grpo`` / ``max_rl`` / ``reward`` / ``custom`` take no teacher.
    Write-only input sugar — folded by validation and excluded from dumps so
    resolved configs round-trip."""

    @model_validator(mode="after")
    def fold_teacher(self):
        """Fold the ``teacher`` shorthand into the slot the algorithm declares.

        Fill-or-agree: the slot (the ``model`` field, or ``sampling.source``
        for source-role algorithms) takes the shorthand when the user didn't
        set it; an explicit reference already equal to it is
        redundant-but-consistent; if no slot accepts it, that's an error."""
        if self.teacher is None:
            return self
        matched = False
        if "model" in type(self).model_fields:
            if self.model is None or "model" not in self.model_fields_set:
                self.model = self.teacher
                matched = True
            elif self.model == self.teacher:
                matched = True
        if getattr(self, "source_role", None) is not None:
            if "source" not in self.sampling.model_fields_set:
                self.sampling.source = self.teacher
                matched = True
            elif self.sampling.source == self.teacher:
                matched = True
        if not matched:
            raise ValueError(
                f"algorithm '{self.type}': 'teacher' is set but the algorithm references no model — "
                "grpo / max_rl / reward / custom take no teacher. Remove it, or use a distillation type."
            )
        return self

    @model_validator(mode="after")
    def validate_references(self):
        source_role = getattr(self, "source_role", None)
        if source_role is not None and self.sampling.source == "policy":
            raise ValueError(
                f"algorithm '{self.type}' needs a {source_role} to sample rollouts from — "
                f"CE on the policy's own tokens is not a distillation target. Set '{source_role}' on "
                "the algorithm (an inline hosted model: name + base_url), or sampling.source explicitly."
            )
        if getattr(self, "model", "<absent>") is None:
            role = getattr(self, "model_role", "reference model")
            raise ValueError(
                f"algorithm '{self.type}' needs a {role} — "
                f"set '{role}' on the algorithm (an inline hosted model: name + base_url), "
                "or set the model directly."
            )
        if isinstance(self, OPDAlgorithmConfig) and self.model == "policy":
            raise ValueError(
                "algorithm 'opd' with model='policy' is degenerate — the reference distribution "
                "equals the policy, so the KL signal is zero. Point at a frozen hosted model, or "
                "use 'opsd' for demo-conditioned self-teaching."
            )
        if self.action_loss_type in ("rl", "ref_kl", "sdpo") and self.sampling.source != "policy":
            raise ValueError(
                f"algorithm '{self.type}' trains with the "
                f"{self.action_loss_type} loss type but sampling.source is a frozen model — "
                "the importance ratio and trust region need the live policy's own sampling logprobs. "
                "Use the 'sft' algorithm to distill frozen-model tokens."
            )
        return self


class GRPOAlgorithmConfig(BaseAlgorithmConfig):
    type: Literal["grpo"] = "grpo"
    """GRPO: scalar advantage = reward minus the per-group mean baseline,
    consumed by the ``rl`` loss component on the rollout's action tokens."""

    action_loss_type: ClassVar[ActionLossType] = "rl"

    length_penalty: LengthPenaltyConfig | None = None
    """Correctness-gated length penalty. ``tokens`` shapes by weighted token cost; ``turns`` shapes by trajectory turn count; None disables shaping. In mixed groups, lower-cost correct rollouts get amplified advantage (up to 2x), higher-cost correct rollouts are unchanged, incorrect untouched. In all-correct groups, below-average-cost rollouts get advantage in [0, 1], others get 0."""


class EchoAlgorithmConfig(GRPOAlgorithmConfig):
    type: Literal["echo"] = "echo"  # type: ignore[assignment]
    """ECHO: group-relative advantage on action tokens (GRPO), plus weighted
    CE on env-provided tokens of later turns (tool output, user feedback),
    selected by message role via the renderer's per-token attribution
    (requires ``orchestrator.renderer``; MITO rollouts carry no attribution).
    Selected tokens feed the ``ce`` loss component at their role's ``alpha``
    and stay outside the rl mask and its denominator."""

    roles: EchoRolesConfig = EchoRolesConfig(tool=EchoRoleConfig())
    """The role table. The default — tool-response bodies at ``alpha = 0.1``
    — is the vetted ECHO setting."""

    filter: EchoFilterConfig | None = None
    """Optional user-supplied filter narrowing the role-selected tokens."""


class MaxRLAlgorithmConfig(BaseAlgorithmConfig):
    type: Literal["max_rl"] = "max_rl"
    """MaxRL (arXiv:2602.02710): scalar advantage = (reward − group mean) /
    group mean, consumed by the ``rl`` loss component. Normalizing by the
    mean instead of GRPO's standard deviation makes the policy gradient
    unbiased for the order-``group_size`` truncation of the maximum-likelihood
    objective: low-pass-rate examples get ~1/p weight, and ``group_size`` is
    the truncation order interpolating REINFORCE (1) → exact maximum
    likelihood (∞). Designed for non-negative (canonically binary) rewards;
    a group with mean reward 0 carries zero advantages everywhere (the
    zero-advantage filter drops it, matching the paper's K=0 convention)."""

    action_loss_type: ClassVar[ActionLossType] = "rl"


class RewardAlgorithmConfig(BaseAlgorithmConfig):
    type: Literal["reward"] = "reward"
    """Scalar advantage = raw reward, no group baseline. Consumed by the
    ``rl`` loss component."""

    action_loss_type: ClassVar[ActionLossType] = "rl"


class OPDAlgorithmConfig(BaseAlgorithmConfig):
    type: Literal["opd"] = "opd"
    """On-policy distillation: the per-token signal is the reverse KL to
    a reference model, evaluated in the trainer from reference prefill
    logprobs scored over each sample's own context (``ref_logprobs`` on the
    wire, ``ref_kl`` loss component). No scalar advantage is assigned —
    rollouts keep ``advantage=None`` (advantage-based filters never fire) and
    samples ship a neutral 0.0; rewards still flow to metrics. ``group_size``
    only fans out sampling."""

    action_loss_type: ClassVar[ActionLossType] = "ref_kl"
    model_role: ClassVar[str] = "teacher"

    model: ModelReference | None = None
    """The teacher — an inline frozen hosted model (``name`` + ``base_url``).
    Required — set it here or via the ``teacher`` shorthand.
    ``"policy"`` is rejected: scoring the policy under itself yields zero KL
    signal (use ``opsd`` for demo-conditioned self-teaching)."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent prefill requests per batch."""


class OPSDAlgorithmConfig(BaseAlgorithmConfig):
    type: Literal["opsd"] = "opsd"
    """On-policy self-distillation (SDFT, https://arxiv.org/abs/2601.19897):
    the per-token signal is the reverse KL to a reference model conditioned on
    an expert demonstration. The scoring prefix is rebuilt from the rollout's
    first-turn messages with the demonstration woven into the user message via
    ``template``; completion logprobs are aligned back onto the sample.
    The default matches the paper's single-step setting; ``multi_turn`` opts
    into scoring each sampled assistant turn in a branch under a rebuilt
    prefix. No scalar advantage is assigned — rollouts keep ``advantage=None``
    (advantage-based filters never fire) and samples ship a neutral 0.0."""

    action_loss_type: ClassVar[ActionLossType] = "ref_kl"
    model_role: ClassVar[str] = "teacher"

    model: ModelReference = "policy"
    """The teacher. ``"policy"`` (the default) is the SDFT paper's setting —
    the current model conditioned on the demo *is* the teacher — and needs no
    extra deployment. Set an inline frozen hosted model to score under a
    frozen copy instead."""

    demo_key: str = "demonstration"
    """Key holding the expert demonstration text — looked up in the example's
    ``info`` dict first, then as a top-level rollout field (e.g. ``answer``)."""

    template: str = (
        "{question}\n\n"
        "Here is an example of an expert response:\n"
        "<demonstration>\n{demonstration}\n</demonstration>\n\n"
        "Answer with a response of your own."
    )
    """Template for the demo-conditioned user message. Receives ``{question}``
    (the original user message text), ``{demonstration}``,
    ``{hindsight_feedback}`` from branch-local observations after the sampled
    turn being scored, and simple scalar values from the rollout ``info`` dict.
    Those extra fields are useful for feedback-conditioned variants that pass
    natural environment or judge feedback alongside the demonstration."""

    template_target: Literal["last_user", "first_user"] = "last_user"
    """Which user message the template rewrites. ``last_user`` preserves the
    original SDFT single-turn behavior; ``first_user`` keeps later user-role
    feedback messages intact in multi-turn agent traces."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent prefill requests per batch."""

    multi_turn: bool = False
    """Opt in to scoring each sampled assistant turn in a multi-turn branch.
    The default keeps OPSD in the SDFT paper's single-step setting."""


class SDPOAlgorithmConfig(BaseAlgorithmConfig):
    type: Literal["sdpo"] = "sdpo"
    """Hübotter-style self-distillation policy optimization. Policy samples
    are re-scored under a feedback-conditioned teacher prompt, and the trainer
    distills over a shared top-k support."""

    action_loss_type: ClassVar[ActionLossType] = "sdpo"
    model_role: ClassVar[str] = "teacher"

    model: ModelReference = "policy"
    """The teacher used for feedback-conditioned scoring. ``"policy"`` keeps
    the original self-distillation setup; a frozen hosted model can be used for
    ``live-policy`` teacher ablations."""

    teacher_regularization: Literal["live-policy", "ema", "trust-region"] = "live-policy"
    """Teacher update semantics. ``live-policy`` scores against the resolved
    teacher reference without a separately maintained teacher. ``ema`` uses a
    separately maintained self-distillation teacher. ``trust-region`` is kept in
    the schema for the loss primitive but is not launchable until a reference
    module source is defined."""

    teacher_update_rate: float = Field(0.05, ge=0, le=1)
    """EMA update rate, or trust-region mixing coefficient."""

    distillation_topk: int = Field(100, ge=1)
    """Top-k support size for SDPO distillation. Combined ``rl`` configs
    validate that every SDPO algorithm matches
    ``trainer.sdpo_loss.distillation_topk``."""

    distillation_topk_support: Literal["teacher", "student"] = "student"
    """Which distribution chooses the top-k support. The Hübotter reference
    uses ``student`` support: the student chooses top-k token ids and the
    teacher is evaluated on those ids. Prime's default ``student`` path runs a
    forward/export-only trainer preflight to produce those ids, then scores the
    teacher on them via vLLM candidate logprobs. ``teacher`` remains available
    as a lighter smoke/ablation mode."""

    preflight_export_timeout_s: int | None = Field(None, gt=0)
    """Optional timeout for waiting on the student-support preflight token export
    STABLE marker. Defaults to unbounded for normal runs; debug smoke presets
    set it so broken export handshakes fail diagnostically."""

    success_reward_threshold: float = 0.5
    """Minimum rollout reward considered successful for selecting
    self-distillation demonstrations."""

    successful_demonstration_selection: Literal["batch_order", "highest_reward"] = "batch_order"
    """How to choose among multiple successful same-task demonstrations.
    ``batch_order`` follows the executable Hübotter/verl reference path by
    taking the first successful sibling in the finalized batch; ``highest_reward``
    is available for shaped-reward Prime/verifiers ablations."""

    dont_reprompt_on_self_success: bool = True
    """When a rollout is already successful, prefer a different successful
    same-task rollout as the demonstration. If none exists, skip the SDPO target
    for that self-success. This follows the active Hübotter/verl training YAML;
    set it to ``False`` for the paper-table behavior where a successful
    original attempt is reprompted with itself as the correct solution."""

    remove_thinking_from_demonstration: bool = True
    """Remove ``<think>...</think>`` spans from successful demonstrations before
    inserting them into the teacher prompt."""

    include_environment_feedback: bool = True
    """Use natural feedback/observations from the branch in failed-sample
    reprompts when no successful demonstration is available."""

    environment_feedback_only_without_solution: bool = True
    """When feedback is available and a successful demonstration is also
    available, omit feedback and condition only on the demonstration."""

    max_reprompt_len: int = Field(10240, ge=1)
    """Maximum length, in tokens, of the rendered teacher reprompt prefix before
    appending the original sampled response."""

    reprompt_truncation: Literal["left", "right", "error"] = "right"
    """How to handle teacher reprompts longer than ``max_reprompt_len``. This
    mirrors the reference tokenizer truncation-side setting."""

    template: str = "{question}{successful_solution_block}{feedback_block}\n\nCorrectly solve the original question."
    """Feedback-conditioned user message template. Receives ``{question}``,
    ``{successful_previous_rollout}``, ``{hindsight_feedback}``, plus
    prebuilt ``{successful_solution_block}`` and ``{feedback_block}``.
    Feedback is omitted when a successful same-question rollout is available,
    matching the paper's preference for a correct solution over failure
    feedback."""

    template_target: Literal["last_user", "first_user"] = "first_user"
    """Which user message the template rewrites. ``first_user`` keeps later
    user-role feedback messages intact in multi-turn traces; ``last_user``
    matches raw prompt-style reprompt construction when the latest user message
    is the original question."""

    solution_template: str = "\nCorrect solution:\n\n{successful_previous_attempt}"
    """Successful-demonstration section template. Receives
    ``{successful_previous_attempt}`` and feeds the outer ``template`` as
    ``{successful_solution_block}``."""

    feedback_template: str = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"
    """Environment-feedback section template. Receives ``{feedback_raw}`` and
    feeds the outer ``template`` as ``{feedback_block}``."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent prefill requests per batch."""

    assistant_prefix: str = ""
    """Optional assistant-side prefix prepended before scoring the original
    sampled response. The reference SDPO config leaves this empty because the
    final instruction is part of the user reprompt template."""

    multi_turn: bool = False
    """Opt in to scoring each sampled assistant turn in a multi-turn branch."""

    @field_validator("teacher_update_rate", "success_reward_threshold", mode="before")
    @classmethod
    def reject_bool_numeric_knobs(cls, value, info):
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be numeric, not boolean")
        return value

    @field_validator(
        "distillation_topk", "preflight_export_timeout_s", "max_reprompt_len", "max_concurrent", mode="before"
    )
    @classmethod
    def reject_bool_integer_knobs(cls, value, info):
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be an integer, not boolean")
        return value

    @field_validator(
        "dont_reprompt_on_self_success",
        "remove_thinking_from_demonstration",
        "include_environment_feedback",
        "environment_feedback_only_without_solution",
        "multi_turn",
        mode="before",
    )
    @classmethod
    def reject_non_bool_behavior_knobs(cls, value, info):
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if not isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a boolean")
        return value

    @model_validator(mode="after")
    def validate_non_live_teacher_regularization_is_internal(self):
        if self.teacher_regularization != "live-policy" and self.model != "policy":
            raise ValueError(
                "sdpo teacher_regularization='ema' or 'trust-region' is an internal "
                "self-distillation teacher mode and must keep model='policy'. External hosted "
                "teacher endpoints are only supported for the live-policy ablation path."
            )
        return self

    @model_validator(mode="after")
    def validate_template_conditioning(self):
        allowed_fields = {
            "question",
            "successful_previous_rollout",
            "hindsight_feedback",
            "successful_solution_block",
            "feedback_block",
        }
        used_fields = _validate_sdpo_format_template(
            self.template,
            allowed_fields,
            name="template",
            fields_message=(
                "question, successful_previous_rollout, hindsight_feedback, successful_solution_block, feedback_block"
            ),
        )
        if "question" not in used_fields:
            raise ValueError("sdpo template must include the {question} field.")
        hindsight_fields = {
            "successful_previous_rollout",
            "hindsight_feedback",
            "successful_solution_block",
            "feedback_block",
        }
        if not used_fields & hindsight_fields:
            raise ValueError(
                "sdpo template must include at least one hindsight conditioning field: "
                "{successful_previous_rollout}, {hindsight_feedback}, {successful_solution_block}, or {feedback_block}."
            )
        return self

    @model_validator(mode="after")
    def validate_section_templates(self):
        solution_fields = _validate_sdpo_format_template(
            self.solution_template,
            {"successful_previous_attempt"},
            name="solution_template",
            fields_message="successful_previous_attempt",
        )
        if "successful_previous_attempt" not in solution_fields:
            raise ValueError("sdpo solution_template must include the {successful_previous_attempt} field.")

        feedback_fields = _validate_sdpo_format_template(
            self.feedback_template,
            {"feedback_raw"},
            name="feedback_template",
            fields_message="feedback_raw",
        )
        if "feedback_raw" not in feedback_fields:
            raise ValueError("sdpo feedback_template must include the {feedback_raw} field.")
        return self


def _validate_sdpo_format_template(
    template: str,
    allowed_fields: set[str],
    *,
    name: str,
    fields_message: str,
) -> set[str]:
    used_fields: set[str] = set()
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"sdpo {name} may only use named fields: {fields_message}.") from exc
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not field_name or "." in field_name or "[" in field_name or field_name not in allowed_fields:
            raise ValueError(f"sdpo {name} may only use named fields: {fields_message}.")
        if conversion is not None:
            raise ValueError(f"sdpo {name} fields may not use conversion flags.")
        if format_spec:
            raise ValueError(f"sdpo {name} fields may not use format specs.")
        used_fields.add(field_name)
    return used_fields


class SFTAlgorithmConfig(BaseAlgorithmConfig):
    type: Literal["sft"] = "sft"
    """SFT distillation: cross-entropy on the sampled tokens. The ``ce`` loss
    ignores advantages and SFT assigns none — it trains on every sampled token.
    Reward-based filtering, if wanted, is an explicit filter, not smuggled
    through an unused advantage stream."""

    action_loss_type: ClassVar[ActionLossType] = "ce"
    source_role: ClassVar[str] = "teacher"
    """The sampling source is this algorithm's teacher — the frozen model
    whose tokens the policy trains on. Required: CE on the policy's own
    tokens is rejected at validation."""


class CustomAlgorithmConfig(BaseAlgorithmConfig):
    type: Literal["custom"] = "custom"
    """Custom advantage function, consumed by the ``rl`` loss component. Returns
    one scalar per rollout, optionally with per-token advantages aligned to
    each rollout's completion tokens."""

    action_loss_type: ClassVar[ActionLossType] = "rl"

    import_path: str
    """Import path to the advantage function (e.g. ``my_module.my_advantage``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the advantage function."""


AlgorithmConfig: TypeAlias = Annotated[
    GRPOAlgorithmConfig
    | EchoAlgorithmConfig
    | MaxRLAlgorithmConfig
    | RewardAlgorithmConfig
    | OPDAlgorithmConfig
    | OPSDAlgorithmConfig
    | SDPOAlgorithmConfig
    | SFTAlgorithmConfig
    | CustomAlgorithmConfig,
    Field(discriminator="type"),
]
"""The training algorithm: sampling plus the per-token training signal (credit
assignment and loss routing, fused). The ``type`` selects the algorithm, and
its class defaults are the vetted setting.

- ``grpo`` — policy group sampling, group-relative advantage, RL loss (the default).
- ``max_rl`` — GRPO with mean-normalized advantages (maximum-likelihood RL).
- ``opd`` — on-policy distillation: policy samples, per-token reverse KL against a reference model. Needs ``teacher``.
- ``opsd`` — SDFT: policy samples, demo-conditioned reverse KL against the live policy by default.
- ``sdpo`` — Hübotter-style SDPO: policy samples, feedback-conditioned top-k self-distillation.
- ``sft`` — a frozen model samples, the policy trains with CE on its tokens. Needs ``teacher``.
- ``echo`` — GRPO on action tokens + weighted CE on tool-response observation tokens.
- ``reward`` / ``custom`` — raw-reward and user-supplied advantage functions.
"""
