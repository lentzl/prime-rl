"""Isolated FlashREINFORCE reference core (paper Eqs. 5-11).

No model, optimizer, collector, communication collective, or Prime-RL hook is installed.
All fragments use a common original-batch trajectory index space. Build the global
plan at fixed learner weights before accumulating any fragment's loss.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryStatistics:
    """Detached, additive statistics indexed by original logical trajectory."""

    token_counts: Tensor
    divergence_sums: Tensor


@dataclass(frozen=True)
class UpdatePlan:
    """Global, detached constants for one fresh-batch optimizer update.

    Treat tensors as immutable. Rebuild after any learner or batch change.
    """

    advantages: Tensor
    token_counts: Tensor
    mean_divergence: Tensor
    accepted: Tensor

    @property
    def batch_size(self) -> int:
        return self.advantages.numel()

    @property
    def has_reward_signal(self) -> bool:
        """Whether any admitted advantage is nonzero, not a gradient-norm test.

        The caller owns optimizer/scheduler skip semantics: zero reward gradient
        alone does not prevent momentum or weight decay from changing weights.
        """
        return bool((self.accepted & (self.advantages != 0)).any())


def _work_dtype(*values: Tensor) -> torch.dtype:
    return torch.float64 if any(x.dtype == torch.float64 for x in values) else torch.float32


def _check_logprobs(values: Tensor, name: str) -> None:
    if not values.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(values).all()) or bool((values > 0).any()):
        raise ValueError(f"selected {name} must be finite log-probabilities <= 0")


def _select_actions(
    current_logprobs: Tensor,
    behavior_logprobs: Tensor,
    trajectory_index: Tensor,
    action_mask: Tensor,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if batch_size < 1:
        raise ValueError("the original batch must contain at least one trajectory")
    arrays = (current_logprobs, behavior_logprobs, trajectory_index, action_mask)
    if any(x.ndim != 1 or x.shape != current_logprobs.shape for x in arrays):
        raise ValueError("logprobs, trajectory_index and action_mask must be aligned 1-D tensors")
    if any(x.device != current_logprobs.device for x in arrays):
        raise ValueError("fragment tensors must be on the same device")
    if trajectory_index.dtype != torch.int64 or action_mask.dtype != torch.bool:
        raise TypeError("trajectory_index must be int64 and action_mask must be bool")
    if not current_logprobs.is_floating_point() or not behavior_logprobs.is_floating_point():
        raise TypeError("logprobs must be floating point")
    # Select first: non-action placeholders may be NaN or have sentinel IDs.
    current = current_logprobs[action_mask]
    behavior = behavior_logprobs[action_mask].detach()
    index = trajectory_index[action_mask]
    if bool(((index < 0) | (index >= batch_size)).any()):
        raise ValueError("selected trajectory indices must lie in the original batch")
    _check_logprobs(current, "current_logprobs")
    _check_logprobs(behavior, "behavior_logprobs")
    dtype = _work_dtype(current, behavior)
    return current.to(dtype), behavior.to(dtype), index


def _log1mexp(negative: Tensor) -> Tensor:
    """log(1-exp(x)) for strictly negative x, without near-zero cancellation."""
    result = torch.empty_like(negative)
    far = negative < -math.log(2.0)
    result[far] = torch.log1p(-torch.exp(negative[far]))
    result[~far] = torch.log(-torch.expm1(negative[~far]))
    return result


@torch.no_grad()
def sampled_action_kl(behavior_logprobs: Tensor, current_logprobs: Tensor) -> Tensor:
    """Bernoulli KL(mu(action) || pi(action)); detached Eq. 6.

    Finite log-probabilities <= 0 are required. Exact probability-one endpoints
    are handled explicitly. A learner probability of one against smaller behavior
    probability gives +inf and is rejected by any finite sequence threshold.
    FP16/BF16 inputs are promoted to FP32; FP64 inputs retain FP64. Promotion cannot
    recover precision that the collector already lost.
    """
    if behavior_logprobs.shape != current_logprobs.shape or behavior_logprobs.device != current_logprobs.device:
        raise ValueError("KL inputs must have matching shapes and devices")
    _check_logprobs(behavior_logprobs, "behavior_logprobs")
    _check_logprobs(current_logprobs, "current_logprobs")
    dtype = _work_dtype(behavior_logprobs, current_logprobs)
    old, new = behavior_logprobs.to(dtype), current_logprobs.to(dtype)
    divergence = torch.zeros_like(old)
    old_one = (old == 0) & (new < 0)
    new_one = (old < 0) & (new == 0)
    interior = (old < 0) & (new < 0) & (old != new)
    divergence[old_one] = -new[old_one]
    divergence[new_one] = float("inf")
    p_log, q_log = old[interior], new[interior]
    divergence[interior] = (
        p_log.exp() * (p_log - q_log)
        + (-p_log.expm1()) * (_log1mexp(p_log) - _log1mexp(q_log))
    )
    if bool(torch.isnan(divergence).any()):
        raise FloatingPointError("nonfinite arithmetic in sampled-action KL")
    return divergence.clamp_min(0)  # Remove negative roundoff, not a divergence cap.


@torch.no_grad()
def trajectory_statistics(
    current_logprobs: Tensor,
    behavior_logprobs: Tensor,
    trajectory_index: Tensor,
    action_mask: Tensor,
    *,
    batch_size: int,
) -> TrajectoryStatistics:
    """Compute local sums/counts; partial trajectories and empty fragments are OK.

    These are not local admission decisions. SUM corresponding vectors across
    all fragments/ranks before calling prepare_update.
    """
    current, behavior, index = _select_actions(
        current_logprobs, behavior_logprobs, trajectory_index, action_mask, batch_size
    )
    counts = torch.bincount(index, minlength=batch_size)
    sums = torch.zeros(batch_size, device=current.device, dtype=current.dtype)
    sums.index_add_(0, index, sampled_action_kl(behavior, current))
    return TrajectoryStatistics(counts, sums)


def _check_statistics(stats: TrajectoryStatistics) -> None:
    counts, sums = stats.token_counts, stats.divergence_sums
    if counts.ndim != 1 or counts.numel() == 0 or counts.shape != sums.shape:
        raise ValueError("statistics must be nonempty, matching trajectory vectors")
    if counts.dtype != torch.int64 or not sums.is_floating_point():
        raise TypeError("statistics need int64 counts and floating-point divergence sums")
    if counts.device != sums.device:
        raise ValueError("statistics must share a device")
    if bool((counts < 0).any()) or bool((sums < 0).any()) or bool(torch.isnan(sums).any()):
        raise ValueError("invalid trajectory statistics")
    if bool(((counts == 0) & (sums != 0)).any()):
        raise ValueError("zero-token statistics cannot contain divergence")


@torch.no_grad()
def merge_statistics(parts: Sequence[TrajectoryStatistics]) -> TrajectoryStatistics:
    """In-process reference for a SUM reduction, not a distributed collective."""
    if not parts:
        raise ValueError("at least one statistics fragment is required")
    first = parts[0]
    for part in parts:
        _check_statistics(part)
        if part.token_counts.shape != first.token_counts.shape or part.token_counts.device != first.token_counts.device:
            raise ValueError("all statistics must use the same batch index space and device")
    counts = first.token_counts.detach().clone()
    sums = first.divergence_sums.detach().clone()
    for part in parts[1:]:
        counts = counts + part.token_counts
        sums = sums + part.divergence_sums
    return TrajectoryStatistics(counts, sums)


@torch.no_grad()
def prepare_update(
    rewards: Tensor,
    statistics: TrajectoryStatistics,
    *,
    threshold: float,
    expected_token_counts: Tensor | None = None,
) -> UpdatePlan:
    """Centre ORIGINAL batch rewards and screen complete trajectories (Eqs. 5-8).

    rewards[i] occurs once per logical trajectory, never once per fragment.
    Missing trajectories raise; optional expected counts detect missing/duplicate
    action positions by count only. Caller still owns action identity/provenance.
    Rejected trajectories remain in both baseline and original batch denominator.
    """
    _check_statistics(statistics)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    counts, sums = statistics.token_counts, statistics.divergence_sums
    if rewards.ndim != 1 or rewards.shape != counts.shape or rewards.device != counts.device:
        raise ValueError("one reward per original trajectory is required on the statistics device")
    if not rewards.is_floating_point() or not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite floating-point values")
    if bool((counts == 0).any()):
        raise ValueError("global statistics omit a trajectory or it has no policy actions")
    if expected_token_counts is not None:
        if (
            expected_token_counts.dtype != torch.int64
            or expected_token_counts.shape != counts.shape
            or expected_token_counts.device != counts.device
        ):
            raise ValueError("expected counts must match the global int64 count vector")
        if not torch.equal(counts, expected_token_counts):
            raise ValueError("action counts differ: missing, duplicated or truncated trajectory data")
    dtype = _work_dtype(rewards, sums)
    values = rewards.to(dtype)
    advantages = values - values.mean()
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("nonfinite centred rewards")
    means = sums.to(dtype) / counts
    return UpdatePlan(advantages, counts.clone(), means, means <= threshold)


def loss_fragment(
    current_logprobs: Tensor,
    behavior_logprobs: Tensor,
    trajectory_index: Tensor,
    action_mask: Tensor,
    plan: UpdatePlan,
) -> Tensor:
    """This fragment's contribution to Eq. 11, already normalized by global B,T.

    SUM all fragment losses/gradients for ONE optimizer update. Do not divide by
    packed-sample/token counts again. Build plan from complete, same-weight
    statistics first; no learner update or changed replay between plan and loss.
    Distributed gradient averaging/compensation belongs to the caller.
    """
    current, behavior, index = _select_actions(
        current_logprobs, behavior_logprobs, trajectory_index, action_mask, plan.batch_size
    )
    if plan.advantages.device != current.device:
        raise ValueError("plan and fragment must share a device")
    if bool((torch.bincount(index, minlength=plan.batch_size) > plan.token_counts).any()):
        raise ValueError("fragment contains more actions than the global plan")
    keep = plan.accepted[index] & (plan.advantages[index] != 0)
    current, behavior, index = current[keep], behavior[keep], index[keep]
    dtype = _work_dtype(current, behavior, plan.advantages)
    current, behavior = current.to(dtype), behavior.to(dtype)
    # No rejected/zero-advantage token is exponentiated or multiplied by infinity.
    with torch.no_grad():
        ratios = (current.detach() - behavior).exp()
        if not bool(torch.isfinite(ratios).all()):
            raise FloatingPointError("admitted importance ratio overflow; no ratio clipping is applied")
        weights = plan.advantages[index] / plan.token_counts[index] / plan.batch_size
    loss = -(weights * ratios * current).sum()
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("nonfinite FlashREINFORCE loss")
    return loss
