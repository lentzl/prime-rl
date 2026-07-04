from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, IPOLossConfig, LossConfig, SDPOComponentConfig
from prime_rl.trainer.rl.sdpo_loss import SDPOLossConfig, compute_sdpo_loss
from prime_rl.trainer.rl.sdpo_train_support import active_sdpo_weight_mask
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample.

    ``loss_mask`` already selects the tokens that belong to the receiving
    component — the component loss functions never re-derive eligibility.
    ``loss_weights`` is the component's per-token weight stream (None means
    1.0 everywhere).
    """

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    ref_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]
    loss_weights: Float[Tensor, " seq"] | None = field(default=None)
    student_topk_log_probs: Float[Tensor, "seq topk"] | None = None
    teacher_topk_log_probs: Float[Tensor, "seq topk"] | None = None
    rollout_is_weights: Float[Tensor, " seq"] | None = None


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]


LossFn = Callable[..., LossOutputs]
"""Type for a per-sample loss function.

Expected signature:
    def my_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
        ...
"""


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(
    logits: Float[Tensor, "batch seq vocab"], left_pad_logit: Float[Tensor, "batch 1 vocab"] | None = None
) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a left pad logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    if left_pad_logit is None:
        left_pad_logit = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([left_pad_logit, logits], dim=1)  # (batch, seq, vocab)
    return logits


def shift_tensor_left(t: Float[Tensor, "batch seq"]) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the left.

    Used to create labels from input_ids: labels[i] = input_ids[i+1].
    The last position is padded with 0 (a valid token index) since this value
    will be shifted off by shift_tensor_right and never used.
    """
    return torch.cat([t[:, 1:], torch.full((t.shape[0], 1), 0, device=t.device, dtype=t.dtype)], dim=1)


def shift_tensor_right(t: Float[Tensor, "batch seq"], pad_value: float | None = None) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the right, prepending a padding value.

    Used to realign logprobs/entropy after computing with shifted labels.
    After shift: result[i] = t[i-1], result[0] = pad_value.
    This converts from "predict next token" convention to "probability of current token" convention.

    Args:
        t: Tensor to shift right
        pad_value: Value to use for position 0. If None, uses 0.0 for backward compatibility.
                   For logprobs, should be log(1/vocab_size) to represent uniform distribution.
                   For entropy, should be log(vocab_size) to represent maximum entropy.
    """
    if pad_value is None:
        pad_value = 0.0
    return torch.cat([torch.full((t.shape[0], 1), pad_value, device=t.device, dtype=t.dtype), t[:, :-1]], dim=1)


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of values over a boolean mask; returns 0 when mask is empty."""
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def compute_importance_ratio_and_mismatch_kl(
    trainer_logprobs: Tensor, inference_logprobs: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1
    return log_importance_ratio, importance_ratio, mismatch_kl


def default_loss_fn(inputs: LossInputs, loss_config: DefaultLossConfig) -> LossOutputs:
    """
    DPPO+KL loss for RL training, combining:
    - DPPO-Binary TV Loss (https://arxiv.org/pdf/2602.04879)
    - Kimi-K2.5 KL Loss (https://arxiv.org/pdf/2602.02276)

    The mask is conditioned on the advantage sign: for positive advantages,
    we mask tokens whose probability increased too much (trust region violation
    in the upweight direction); for negative advantages, we mask tokens whose
    probability decreased too much (trust region violation in the downweight
    direction).
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    dppo_invalid_mask_high = probs_diff > loss_config.dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -loss_config.dppo_mask_low
    positive_advantages = advantages > 0
    negative_advantages = advantages < 0
    dppo_invalid_mask = torch.where(positive_advantages, dppo_invalid_mask_high, dppo_invalid_mask_low)

    is_masked = dppo_invalid_mask
    is_masked_high = positive_advantages & dppo_invalid_mask_high
    is_masked_low = negative_advantages & dppo_invalid_mask_low
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    advantages = loss_config.adv_tau * advantages
    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    per_token_loss = -pg_loss + loss_config.kl_tau * kl_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
        "masked_advantage_positive": _safe_mean(positive_advantages, drop_mask),
        "masked_advantage_negative": _safe_mean(negative_advantages, drop_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def ipo_loss_fn(inputs: LossInputs, loss_config: IPOLossConfig) -> LossOutputs:
    """IPO loss type: a symmetric trust region (mask tokens whose probability
    moved more than ``ipo_threshold`` in absolute terms), policy gradient via
    the importance ratio, and a squared-log-ratio KL regularizer."""
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    abs_probs_diff = torch.abs(torch.exp(trainer_logprobs) - torch.exp(inference_logprobs))

    is_masked = abs_probs_diff > loss_config.ipo_threshold
    keep_mask = loss_mask & ~is_masked

    advantages = loss_config.adv_tau * advantages
    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    per_token_loss = -pg_loss + loss_config.kl_tau * kl_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()

    metrics = {
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def ref_kl_loss_fn(inputs: LossInputs) -> LossOutputs:
    """
    Ref-KL loss type (on-policy distillation): the reverse KL to the reference
    model is the per-token policy-gradient signal, with the importance ratio
    correcting trainer/inference mismatch and staleness. A one-sided trust
    region drops tokens whose trainer probability fell more than 0.2 below the
    inference probability; a squared-log-ratio term regularizes drift. Scalar
    advantages are not read — ref_kl algorithms ship none.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    ref_logprobs = inputs.ref_logprobs
    loss_mask = inputs.loss_mask

    if ref_logprobs is None:
        raise ValueError("ref_kl loss type requires ref_logprobs — use an 'opd' or 'opsd' advantage strategy.")

    log_importance_ratio, importance_ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        trainer_logprobs, inference_logprobs
    )

    probs_diff = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
    is_masked = probs_diff < -0.2
    drop_mask = loss_mask & is_masked
    keep_mask = loss_mask & ~is_masked

    ref_kl = ref_logprobs - trainer_logprobs

    pg_loss = keep_mask * ref_kl.detach() * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    per_token_loss = -pg_loss + 1e-3 * kl_loss
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()

    # Namespaced: the rl loss fn emits same-named trust-region metrics with a
    # different definition, and mixed batches run both fns in one step.
    metrics = {
        "ref_kl/masked_mismatch_kl": _safe_mean(mismatch_kl, drop_mask),
        "ref_kl/unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),
        "ref_kl/is_masked": _safe_mean(is_masked, loss_mask),
        "ref_kl": _safe_mean(ref_kl, loss_mask),
    }

    return LossOutputs(loss=loss, metrics=metrics)


def sdpo_loss_fn(inputs: LossInputs, loss_config: SDPOComponentConfig) -> LossOutputs:
    """SDPO over the component-selected tokens."""
    if loss_config.full_logit_distillation and loss_config.distillation_topk is None:
        raise ValueError("SDPO full-vocabulary distillation is not wired yet; set distillation_topk.")
    if loss_config.full_logit_distillation:
        if inputs.student_topk_log_probs is None or inputs.teacher_topk_log_probs is None:
            raise ValueError("SDPO top-k loss requires student and teacher top-k logprobs.")
        if inputs.student_topk_log_probs.shape != inputs.teacher_topk_log_probs.shape:
            raise ValueError(
                "SDPO top-k student and teacher logprob shapes must match "
                f"({tuple(inputs.student_topk_log_probs.shape)} != {tuple(inputs.teacher_topk_log_probs.shape)})."
            )
    elif inputs.ref_logprobs is None:
        raise ValueError("sampled-token SDPO requires ref_logprobs.")

    active_mask = inputs.loss_mask
    if inputs.loss_weights is not None:
        active_mask = active_mask & _active_sdpo_member_mask(inputs.loss_weights)
    token_count = active_mask.sum().clamp(min=1)
    if not bool(active_mask.any()):
        zero = inputs.trainer_logprobs.sum() * 0.0
        return LossOutputs(loss=zero, metrics={"sdpo": zero.detach()})

    trainer_logprobs = inputs.trainer_logprobs[active_mask]
    inference_logprobs = inputs.inference_logprobs[active_mask]
    ref_logprobs = inputs.ref_logprobs[active_mask] if inputs.ref_logprobs is not None else None
    student_topk_log_probs = (
        inputs.student_topk_log_probs[active_mask] if inputs.student_topk_log_probs is not None else None
    )
    teacher_topk_log_probs = (
        inputs.teacher_topk_log_probs[active_mask] if inputs.teacher_topk_log_probs is not None else None
    )
    loss_weights = inputs.loss_weights[active_mask] if inputs.loss_weights is not None else None
    rollout_is_weights = inputs.rollout_is_weights[active_mask] if inputs.rollout_is_weights is not None else None
    response_mask = torch.ones_like(trainer_logprobs, dtype=torch.bool)

    core_config = SDPOLossConfig(
        full_logit_distillation=loss_config.full_logit_distillation,
        distillation_topk=loss_config.distillation_topk,
        distillation_add_tail=loss_config.distillation_add_tail,
        alpha=loss_config.alpha,
        is_clip=loss_config.is_clip,
        rollout_is=loss_config.rollout_is,
        rollout_is_threshold=loss_config.rollout_is_threshold,
        rollout_is_batch_normalize=loss_config.rollout_is_batch_normalize,
    )
    old_log_probs = (
        inference_logprobs.unsqueeze(0)
        if loss_config.is_clip is not None or (rollout_is_weights is None and loss_config.rollout_is is not None)
        else None
    )
    loss = compute_sdpo_loss(
        student_log_probs=trainer_logprobs.unsqueeze(0),
        teacher_log_probs=(ref_logprobs if ref_logprobs is not None else trainer_logprobs).detach().unsqueeze(0),
        response_mask=response_mask.unsqueeze(0),
        config=core_config,
        old_log_probs=old_log_probs,
        student_topk_log_probs=student_topk_log_probs.unsqueeze(0) if student_topk_log_probs is not None else None,
        teacher_topk_log_probs=teacher_topk_log_probs.detach().unsqueeze(0)
        if teacher_topk_log_probs is not None
        else None,
        component_weights=loss_weights.unsqueeze(0) if loss_weights is not None else None,
        rollout_is_weights=rollout_is_weights.unsqueeze(0) if rollout_is_weights is not None else None,
    )
    metrics = {"sdpo": loss.detach()}
    return LossOutputs(loss=loss * token_count, metrics=metrics)


def _validate_sdpo_component_weights(sdpo_weights: Tensor, loss_mask: Tensor) -> None:
    if sdpo_weights.shape != loss_mask.shape:
        raise ValueError(
            f"SDPO weights shape {tuple(sdpo_weights.shape)} must match loss_mask shape {tuple(loss_mask.shape)}"
        )
    if torch.is_complex(sdpo_weights) or sdpo_weights.dtype == torch.bool:
        raise ValueError("SDPO weights must contain real numeric values")
    if not bool(torch.isfinite(sdpo_weights).all()):
        raise ValueError("SDPO weights must contain finite values")
    if bool((sdpo_weights < 0).any()):
        raise ValueError("SDPO weights must be non-negative")
    invalid_sdpo_mask = _active_sdpo_member_mask(sdpo_weights) & ~loss_mask
    if bool(invalid_sdpo_mask.any()):
        raise ValueError("SDPO weights may only select sampled/loss-mask tokens")


def _validate_sdpo_rollout_is_weights(
    rollout_is_weights: Tensor,
    sdpo_weights: Tensor | None,
    loss_mask: Tensor,
) -> None:
    if rollout_is_weights.shape != loss_mask.shape:
        raise ValueError(
            "SDPO rollout-IS weights shape "
            f"{tuple(rollout_is_weights.shape)} must match loss_mask shape {tuple(loss_mask.shape)}"
        )
    if torch.is_complex(rollout_is_weights) or rollout_is_weights.dtype == torch.bool:
        raise ValueError("SDPO rollout-IS weights must contain real numeric values")
    if not bool(torch.isfinite(rollout_is_weights).all()):
        raise ValueError("SDPO rollout-IS weights must contain finite values")
    if bool((rollout_is_weights < 0).any()):
        raise ValueError("SDPO rollout-IS weights must be non-negative")
    if not torch.is_floating_point(rollout_is_weights):
        raise ValueError("SDPO rollout-IS weights must use a floating-point dtype")
    sdpo_member_mask = (
        torch.zeros_like(loss_mask, dtype=torch.bool)
        if sdpo_weights is None
        else _active_sdpo_member_mask(sdpo_weights)
    )
    invalid_rollout_is_mask = _active_sdpo_member_mask(rollout_is_weights) & ~sdpo_member_mask
    if bool(invalid_rollout_is_mask.any()):
        raise ValueError("SDPO rollout-IS weights may only be nonzero on SDPO component tokens")


def _validate_sdpo_topk_ownership(
    student_topk_log_probs: Tensor | None,
    teacher_topk_log_probs: Tensor | None,
    sdpo_weights: Tensor | None,
    loss_mask: Tensor,
) -> None:
    if student_topk_log_probs is None and teacher_topk_log_probs is None:
        return
    if student_topk_log_probs is None or teacher_topk_log_probs is None:
        raise ValueError("SDPO top-k logprobs require both student and teacher streams")
    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "SDPO top-k student and teacher logprob shapes must match "
            f"({tuple(student_topk_log_probs.shape)} != {tuple(teacher_topk_log_probs.shape)})."
        )
    if student_topk_log_probs.ndim != 2:
        raise ValueError(
            f"SDPO top-k logprobs must be rank 2 per sequence (seq, topk), got {tuple(student_topk_log_probs.shape)}"
        )
    if student_topk_log_probs.shape[0] != loss_mask.shape[0]:
        raise ValueError(
            "SDPO top-k logprobs sequence length "
            f"{student_topk_log_probs.shape[0]} must match loss_mask length {loss_mask.shape[0]}"
        )
    if student_topk_log_probs.shape[1] <= 0:
        raise ValueError("SDPO top-k logprobs must have a non-empty top-k dimension")
    if sdpo_weights is None:
        raise ValueError("SDPO top-k logprobs require active SDPO weights")
    active_sdpo_mask = loss_mask & _active_sdpo_member_mask(sdpo_weights)
    if not bool(active_sdpo_mask.any()):
        raise ValueError("SDPO top-k logprobs require active SDPO weights")
    inactive_topk_mask = ~active_sdpo_mask
    inactive_student = student_topk_log_probs[inactive_topk_mask]
    inactive_teacher = teacher_topk_log_probs[inactive_topk_mask]
    invalid_inactive_rows = ~((inactive_student == 0).all(dim=-1) & (inactive_teacher == 0).all(dim=-1))
    if bool(invalid_inactive_rows.any()):
        raise ValueError("SDPO top-k logprobs must be placeholders outside active SDPO component")


def _active_sdpo_member_mask(weights: Tensor) -> Tensor:
    return active_sdpo_weight_mask(weights)


def ce_loss_fn(inputs: LossInputs) -> LossOutputs:
    """Cross-entropy loss type: masked negative log-likelihood (SFT / ECHO
    observation prediction)."""
    trainer_logprobs = inputs.trainer_logprobs
    loss_mask = inputs.loss_mask

    nll = -trainer_logprobs
    if inputs.loss_weights is not None:
        nll = nll * inputs.loss_weights
    loss = nll[loss_mask].sum()
    metrics = {
        "nll": _safe_mean(-trainer_logprobs, loss_mask),
    }
    return LossOutputs(loss=loss, metrics=metrics)


def setup_rl_loss_fn(loss_config: LossConfig) -> LossFn:
    """Build the loss fn for the rl component from ``trainer.loss``:
    ``default_loss_fn`` (``DefaultLossConfig``), ``ipo_loss_fn``
    (``IPOLossConfig``), or the imported function (``CustomLossConfig``).
    The ce / ref_kl loss types are fixed and unaffected by ``trainer.loss``."""
    if isinstance(loss_config, CustomLossConfig):
        custom_fn = import_object(loss_config.import_path)
        kwargs = loss_config.kwargs

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return custom_fn(inputs, **kwargs)
    elif isinstance(loss_config, IPOLossConfig):

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return ipo_loss_fn(inputs, loss_config)
    else:

        def rl_fn(inputs: LossInputs) -> LossOutputs:
            return default_loss_fn(inputs, loss_config)

    return rl_fn


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    ref_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    rl_weights: list[Float[Tensor, " seq_i"]] | None,
    ce_weights: list[Float[Tensor, " seq_i"]] | None,
    ref_kl_weights: list[Float[Tensor, " seq_i"]] | None,
    rl_loss_fn: LossFn,
    rl_scale: int,
    ce_scale: int,
    ref_kl_scale: int,
    sdpo_weights: list[Float[Tensor, " seq_i"]] | None = None,
    sdpo_rollout_is_weights: list[Float[Tensor, " seq_i"]] | None = None,
    student_topk_log_probs: list[Float[Tensor, "seq_i topk"]] | None = None,
    teacher_topk_log_probs: list[Float[Tensor, "seq_i topk"]] | None = None,
    sdpo_loss_config: SDPOComponentConfig | None = None,
    sdpo_scale: int = 1,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    The loss is a sum of four components, each running over its own per-token
    weight stream and normalized by its own global token count:

    - rl → ``rl_loss_fn`` (built by ``setup_rl_loss_fn``) on
      ``loss_mask & (rl_weights != 0)``; an absent stream means weight 1.0 on
      the full loss mask (the hot path — no extra device syncs).
    - ce → ``ce_loss_fn`` (masked NLL) on ``ce_weights != 0``.
    - ref_kl → ``ref_kl_loss_fn`` on ``ref_kl_weights != 0``.
    - sdpo → ``sdpo_loss_fn`` on the active ``sdpo_weights`` membership mask.

    A weight scales its component's per-token loss; 0.0 removes the token from
    the component's mask and denominator. Per-component normalization keeps the
    components from diluting each other: a token only enters the denominator of
    the components it belongs to.

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Sampling-policy log probabilities for each sequence
        ref_logprobs: Reference-model log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        rl_weights: Per-token rl weights for each sequence, or None (1.0 on the loss mask)
        ce_weights: Per-token ce weights for each sequence, or None (no ce component)
        ref_kl_weights: Per-token ref_kl weights for each sequence, or None (no ref_kl component)
        sdpo_weights: Per-token sdpo weights for each sequence, or None (no sdpo component)
        sdpo_rollout_is_weights: Optional true rollout-importance weights for SDPO.
        rl_loss_fn: Loss fn for the rl component from setup_rl_loss_fn()
        rl_scale: Global rl-token count normalizing the rl component
        ce_scale: Global ce-token count normalizing the ce component
        ref_kl_scale: Global ref_kl-token count normalizing the ref_kl component
        sdpo_scale: Global sdpo-token count normalizing the sdpo component

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)
    """
    all_metrics: dict[str, list[Tensor]] = {}
    if sdpo_loss_config is None:
        sdpo_loss_config = SDPOComponentConfig()

    n = len(trainer_logprobs)
    for name, values in (
        ("inference_logprobs", inference_logprobs),
        ("advantages", advantages),
        ("loss_mask", loss_mask),
    ):
        if len(values) != n:
            raise ValueError(f"{name} has {len(values)} sequence(s), expected {n}")
    for name, values in (
        ("ref_logprobs", ref_logprobs),
        ("rl_weights", rl_weights),
        ("ce_weights", ce_weights),
        ("ref_kl_weights", ref_kl_weights),
        ("sdpo_weights", sdpo_weights),
        ("sdpo_rollout_is_weights", sdpo_rollout_is_weights),
        ("student_topk_log_probs", student_topk_log_probs),
        ("teacher_topk_log_probs", teacher_topk_log_probs),
    ):
        if values is not None and len(values) != n:
            raise ValueError(f"{name} has {len(values)} sequence(s), expected {n}")
    if ref_logprobs is None:
        ref_logprobs = [None] * n
    if rl_weights is None:
        rl_weights = [None] * n
    if ce_weights is None:
        ce_weights = [None] * n
    if ref_kl_weights is None:
        ref_kl_weights = [None] * n
    if sdpo_weights is None:
        sdpo_weights = [None] * n
    if sdpo_rollout_is_weights is None:
        sdpo_rollout_is_weights = [None] * n
    if student_topk_log_probs is None:
        student_topk_log_probs = [None] * n
    if teacher_topk_log_probs is None:
        teacher_topk_log_probs = [None] * n

    def run_loss_fn(loss_fn: LossFn, inputs: LossInputs) -> Tensor:
        result = loss_fn(inputs)
        for k, v in result.metrics.items():
            all_metrics.setdefault(k, []).append(v)
        return result.loss

    # Graph anchor: a micro batch whose components are all empty (e.g. a fully
    # truncated distillation sample, whose stamped streams survive as all-zero
    # prefixes) must still return a backward-able loss so every rank runs
    # backward and FSDP collectives stay in sync.
    rl_loss = trainer_logprobs[0].sum() * 0.0
    ce_loss = 0.0
    ref_kl_loss = 0.0
    sdpo_loss = 0.0
    for (
        t_logp,
        i_logp,
        ref_logp,
        student_topk,
        teacher_topk,
        adv,
        mask,
        rl_w,
        ce_w,
        ref_kl_w,
        sdpo_w,
        sdpo_is_w,
    ) in zip(
        trainer_logprobs,
        inference_logprobs,
        ref_logprobs,
        student_topk_log_probs,
        teacher_topk_log_probs,
        advantages,
        loss_mask,
        rl_weights,
        ce_weights,
        ref_kl_weights,
        sdpo_weights,
        sdpo_rollout_is_weights,
    ):

        def make_inputs(component_mask: Bool[Tensor, " seq"], weights: Float[Tensor, " seq"] | None) -> LossInputs:
            return LossInputs(
                trainer_logprobs=t_logp,
                inference_logprobs=i_logp,
                ref_logprobs=ref_logp,
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                advantages=adv,
                loss_mask=component_mask,
                loss_weights=weights,
                rollout_is_weights=sdpo_is_w,
            )

        if rl_w is None:
            rl_loss = rl_loss + run_loss_fn(rl_loss_fn, make_inputs(mask, None))
        else:
            rl_mask = mask & (rl_w != 0)
            if bool(rl_mask.any()):
                rl_loss = rl_loss + run_loss_fn(rl_loss_fn, make_inputs(rl_mask, rl_w))
        if ce_w is not None:
            ce_mask = ce_w != 0
            if bool(ce_mask.any()):
                ce_loss = ce_loss + run_loss_fn(ce_loss_fn, make_inputs(ce_mask, ce_w))
        if ref_kl_w is not None:
            ref_kl_mask = ref_kl_w != 0
            if bool(ref_kl_mask.any()):
                ref_kl_loss = ref_kl_loss + run_loss_fn(ref_kl_loss_fn, make_inputs(ref_kl_mask, ref_kl_w))
        if sdpo_w is not None:
            _validate_sdpo_component_weights(sdpo_w, mask)
        if sdpo_is_w is not None:
            _validate_sdpo_rollout_is_weights(sdpo_is_w, sdpo_w, mask)
        _validate_sdpo_topk_ownership(student_topk, teacher_topk, sdpo_w, mask)
        if sdpo_w is not None:
            sdpo_mask = mask & _active_sdpo_member_mask(sdpo_w)
            if bool(sdpo_mask.any()):
                if (
                    sdpo_is_w is None
                    and sdpo_loss_config.rollout_is is not None
                    and sdpo_loss_config.rollout_is_batch_normalize
                ):
                    raise ValueError(
                        "compute_loss cannot derive batch-normalized SDPO rollout-IS weights per packed sequence; "
                        "provide precomputed sdpo_rollout_is_weights or disable rollout_is_batch_normalize."
                    )
                sdpo_loss = sdpo_loss + run_loss_fn(
                    lambda inputs: sdpo_loss_fn(inputs, sdpo_loss_config), make_inputs(sdpo_mask, sdpo_w)
                )

    scaled_loss = rl_loss / rl_scale + ce_loss / ce_scale + ref_kl_loss / ref_kl_scale + sdpo_loss / sdpo_scale

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    return scaled_loss, aggregated
