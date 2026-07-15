from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class SDPOLossConfig:
    full_logit_distillation: bool = True
    distillation_topk: int | None = 100
    distillation_add_tail: bool = True
    alpha: float = 0.5
    is_clip: float | None = None
    rollout_is: str | None = None
    rollout_is_threshold: float = 2.0
    rollout_is_batch_normalize: bool = False

    def __post_init__(self) -> None:
        for name in ("full_logit_distillation", "distillation_add_tail", "rollout_is_batch_normalize"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"SDPO {name} must be a boolean")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)) or not math.isfinite(self.alpha):
            raise ValueError("SDPO alpha must be a finite number")
        if not 0 <= self.alpha <= 1:
            raise ValueError("SDPO alpha must be in [0, 1]")
        if self.distillation_topk is not None:
            if isinstance(self.distillation_topk, bool) or not isinstance(self.distillation_topk, int):
                raise ValueError("SDPO distillation_topk must be an integer when set")
            if self.distillation_topk <= 0:
                raise ValueError("SDPO distillation_topk must be positive when set")
        if self.is_clip is not None:
            if (
                isinstance(self.is_clip, bool)
                or not isinstance(self.is_clip, (int, float))
                or not math.isfinite(self.is_clip)
            ):
                raise ValueError("SDPO is_clip must be a finite number when set")
            if self.is_clip <= 0:
                raise ValueError("SDPO is_clip must be positive when set")
        if self.rollout_is is not None and self.rollout_is not in {"token", "sequence"}:
            raise ValueError("SDPO rollout_is must be 'token', 'sequence', or None")
        if (
            isinstance(self.rollout_is_threshold, bool)
            or not isinstance(self.rollout_is_threshold, (int, float))
            or not math.isfinite(self.rollout_is_threshold)
        ):
            raise ValueError("SDPO rollout_is_threshold must be a finite number")
        if self.rollout_is_threshold <= 0:
            raise ValueError("SDPO rollout_is_threshold must be positive")


def _add_tail(log_probs: Tensor) -> Tensor:
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _renorm_topk_log_probs(log_probs: Tensor) -> Tensor:
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def _require_log_probability_rows(log_probs: Tensor, name: str, *, row_mask: Tensor | None = None) -> None:
    _require_floating_finite_values(log_probs, name, value_name="log-probabilities")
    if row_mask is not None:
        _require_shape(row_mask, log_probs.shape[:-1], f"{name} row_mask")
        _require_bool_mask(row_mask, f"{name} row_mask")
        if not bool(row_mask.any()):
            return
        log_probs = log_probs[row_mask]
    row_log_mass = torch.logsumexp(log_probs, dim=-1)
    if bool((row_log_mass > 1e-5).any()):
        raise ValueError(f"{name} rows must have probability mass <= 1")


def _require_finite_values(tensor: Tensor, name: str) -> None:
    if torch.is_complex(tensor) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite values")


def _require_floating_finite_values(tensor: Tensor, name: str, *, value_name: str = "values") -> None:
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite {value_name}")


def _require_nonnegative_finite_weights(weights: Tensor, name: str) -> None:
    if weights.dtype == torch.bool or torch.is_complex(weights):
        raise ValueError(f"{name} must contain finite non-negative numeric weights")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError(f"{name} must contain finite weights")
    if bool((weights < 0).any()):
        raise ValueError(f"{name} must contain non-negative weights")


def _require_nonnegative_floating_finite_weights(weights: Tensor, name: str) -> None:
    _require_nonnegative_finite_weights(weights, name)
    if not torch.is_floating_point(weights):
        raise ValueError(f"{name} must use a floating-point dtype")


def _require_weights_at_most(weights: Tensor, threshold: float, name: str) -> None:
    if bool((weights > threshold + 1e-6).any()):
        raise ValueError(f"{name} must not exceed rollout_is_threshold={threshold}")


def _require_bool_mask(mask: Tensor, name: str) -> None:
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean mask")


def _require_binary_mask_values(mask: Tensor, name: str) -> None:
    if mask.dtype == torch.bool:
        return
    _require_finite_values(mask, name)
    if bool(((mask != 0) & (mask != 1)).any()):
        raise ValueError(f"{name} must contain boolean/binary mask values")


def _token_mean(per_token_loss: Tensor, loss_mask: Tensor) -> Tensor:
    return (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)


def _require_shape(tensor: Tensor, expected: torch.Size, name: str) -> None:
    if tensor.shape != expected:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} must match {tuple(expected)}")


def compute_rollout_is_weights(
    *,
    log_ratio: Tensor,
    response_mask: Tensor,
    rollout_is: str,
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> Tensor:
    if rollout_is not in {"token", "sequence"}:
        raise ValueError("rollout_is must be 'token' or 'sequence'")
    if (
        isinstance(rollout_is_threshold, bool)
        or not isinstance(rollout_is_threshold, (int, float))
        or not math.isfinite(float(rollout_is_threshold))
    ):
        raise ValueError("rollout_is_threshold must be a finite number")
    if rollout_is_threshold <= 0:
        raise ValueError("rollout_is_threshold must be positive")
    if not isinstance(rollout_is_batch_normalize, bool):
        raise ValueError("rollout_is_batch_normalize must be a boolean")
    _require_shape(response_mask, log_ratio.shape, "response_mask")
    _require_floating_finite_values(log_ratio, "log_ratio")
    _require_binary_mask_values(response_mask, "response_mask")

    response_mask = response_mask.to(log_ratio.dtype)
    if rollout_is == "token":
        weights = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
    else:
        seq_log_ratio = (log_ratio * response_mask).sum(dim=-1, keepdim=True)
        weights = torch.exp(torch.clamp(seq_log_ratio, min=-20.0, max=20.0)).expand_as(log_ratio)

    weights = weights * response_mask
    weights = weights.clamp(max=rollout_is_threshold).detach()
    if rollout_is_batch_normalize:
        if rollout_is == "token":
            denom = response_mask.sum().clamp(min=1.0)
            mean = (weights * response_mask).sum() / denom
        else:
            seq_mask = response_mask.sum(dim=-1) > 0
            seq_weights = (weights * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1.0)
            mean = seq_weights[seq_mask].sum() / seq_mask.sum().clamp(min=1)
        if bool(mean > 1e-8):
            weights = weights / mean
    return weights


def compute_sdpo_loss(
    *,
    student_log_probs: Tensor,
    teacher_log_probs: Tensor,
    response_mask: Tensor,
    config: SDPOLossConfig,
    old_log_probs: Tensor | None = None,
    student_all_log_probs: Tensor | None = None,
    teacher_all_log_probs: Tensor | None = None,
    student_topk_log_probs: Tensor | None = None,
    teacher_topk_log_probs: Tensor | None = None,
    self_distillation_mask: Tensor | None = None,
    component_weights: Tensor | None = None,
    rollout_is_weights: Tensor | None = None,
) -> Tensor:
    """Compute the SDPO distillation loss.

    Inputs are batch-shaped tensors over response positions; full-logit inputs
    add a final vocabulary/top-k dimension.
    """

    _require_bool_mask(response_mask, "response_mask")
    loss_mask = response_mask.to(student_log_probs.dtype)
    _require_shape(teacher_log_probs, student_log_probs.shape, "teacher_log_probs")
    _require_shape(loss_mask, student_log_probs.shape, "response_mask")
    _require_floating_finite_values(student_log_probs, "student_log_probs")
    _require_floating_finite_values(teacher_log_probs, "teacher_log_probs")
    if self_distillation_mask is not None:
        _require_bool_mask(self_distillation_mask, "self_distillation_mask")
        if self_distillation_mask.shape != student_log_probs.shape[:1]:
            raise ValueError(
                f"self_distillation_mask shape {tuple(self_distillation_mask.shape)} must match "
                f"batch shape {tuple(student_log_probs.shape[:1])}"
            )
        loss_mask = loss_mask * self_distillation_mask.to(student_log_probs.dtype).unsqueeze(1)
    if component_weights is not None:
        _require_shape(component_weights, student_log_probs.shape, "component_weights")
        _require_nonnegative_finite_weights(component_weights, "component_weights")
        loss_mask = loss_mask * (component_weights != 0).to(student_log_probs.dtype)

    if config.full_logit_distillation:
        use_topk = config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError("top-k SDPO requires student_topk_log_probs and teacher_topk_log_probs")
            if student_topk_log_probs.shape[:2] != student_log_probs.shape:
                raise ValueError(
                    f"student_topk_log_probs leading shape {tuple(student_topk_log_probs.shape[:2])} "
                    f"must match student_log_probs shape {tuple(student_log_probs.shape)}"
                )
            _require_shape(teacher_topk_log_probs, student_topk_log_probs.shape, "teacher_topk_log_probs")
            expected_ndim = student_log_probs.ndim + 1
            if student_topk_log_probs.ndim != expected_ndim:
                raise ValueError(
                    f"student_topk_log_probs rank {student_topk_log_probs.ndim} must be {expected_ndim} "
                    "with a final top-k dimension"
                )
            if student_topk_log_probs.shape[-1] != config.distillation_topk:
                raise ValueError(
                    f"student_topk_log_probs width {student_topk_log_probs.shape[-1]} must match "
                    f"distillation_topk={config.distillation_topk}"
                )
            distillation_row_mask = loss_mask.to(torch.bool)
            _require_log_probability_rows(
                student_topk_log_probs,
                "student_topk_log_probs",
                row_mask=distillation_row_mask,
            )
            _require_log_probability_rows(
                teacher_topk_log_probs,
                "teacher_topk_log_probs",
                row_mask=distillation_row_mask,
            )
            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if config.distillation_add_tail:
                student_distill_log_probs = _add_tail(student_distill_log_probs)
                teacher_distill_log_probs = _add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = _renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = _renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError("full-logit SDPO requires student_all_log_probs and teacher_all_log_probs")
            if student_all_log_probs.shape[:2] != student_log_probs.shape:
                raise ValueError(
                    f"student_all_log_probs leading shape {tuple(student_all_log_probs.shape[:2])} "
                    f"must match student_log_probs shape {tuple(student_log_probs.shape)}"
                )
            _require_shape(teacher_all_log_probs, student_all_log_probs.shape, "teacher_all_log_probs")
            distillation_row_mask = loss_mask.to(torch.bool)
            _require_log_probability_rows(
                student_all_log_probs,
                "student_all_log_probs",
                row_mask=distillation_row_mask,
            )
            _require_log_probability_rows(
                teacher_all_log_probs,
                "teacher_all_log_probs",
                row_mask=distillation_row_mask,
            )
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        if config.alpha == 0.0:
            kl_loss = F.kl_div(student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
        elif config.alpha == 1.0:
            kl_loss = F.kl_div(teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True)
        else:
            alpha = torch.tensor(
                config.alpha, dtype=student_distill_log_probs.dtype, device=student_distill_log_probs.device
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_distill_log_probs + torch.log1p(-alpha),
                        teacher_distill_log_probs + torch.log(alpha),
                    ]
                ),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha)

        per_token_loss = kl_loss.sum(-1)
    else:
        if config.alpha != 1.0:
            raise ValueError("sampled-token SDPO only supports alpha=1.0")
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs

    if config.is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required when SDPO is_clip is enabled")
        _require_shape(old_log_probs, student_log_probs.shape, "old_log_probs")
        _require_floating_finite_values(old_log_probs, "old_log_probs")
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=config.is_clip)
        per_token_loss = per_token_loss * ratio

    if rollout_is_weights is None and config.rollout_is is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required when SDPO rollout_is is enabled")
        _require_shape(old_log_probs, student_log_probs.shape, "old_log_probs")
        _require_floating_finite_values(old_log_probs, "old_log_probs")
        rollout_is_weights = compute_rollout_is_weights(
            log_ratio=(student_log_probs - old_log_probs).detach(),
            response_mask=loss_mask,
            rollout_is=config.rollout_is,
            rollout_is_threshold=config.rollout_is_threshold,
            rollout_is_batch_normalize=config.rollout_is_batch_normalize,
        )

    if rollout_is_weights is not None:
        _require_shape(rollout_is_weights, student_log_probs.shape, "rollout_is_weights")
        _require_nonnegative_floating_finite_weights(rollout_is_weights, "rollout_is_weights")
        if config.rollout_is is not None and not config.rollout_is_batch_normalize:
            _require_weights_at_most(
                rollout_is_weights,
                float(config.rollout_is_threshold),
                "rollout_is_weights",
            )
        per_token_loss = per_token_loss * rollout_is_weights
    if component_weights is not None:
        per_token_loss = per_token_loss * component_weights

    return _token_mean(per_token_loss, loss_mask)
