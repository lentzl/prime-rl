from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


def resolve_preflight_batch_mode(micro_batches: Sequence[Mapping[str, Any]], *, enable_token_export: bool) -> bool:
    """Return whether a trainer batch is forward/export-only.

    SDPO student-support preflight is only valid as a whole-step mode: every
    micro-batch must agree, and it needs token export so the orchestrator can
    hydrate the student-selected support before the final training batch.
    """
    if not micro_batches:
        raise ValueError("Cannot resolve preflight mode for an empty trainer batch")
    preflight_flags: set[bool] = set()
    for idx, micro_batch in enumerate(micro_batches):
        if "preflight_only" not in micro_batch:
            raise ValueError(f"preflight_only must be present on every micro batch (missing at index {idx})")
        preflight_only = micro_batch["preflight_only"]
        if not isinstance(preflight_only, bool):
            raise ValueError("preflight_only must be a boolean on every micro batch")
        preflight_flags.add(preflight_only)
    if len(preflight_flags) != 1:
        raise ValueError("Cannot mix preflight-only and train micro batches in one trainer step")
    preflight_only = preflight_flags.pop()
    if preflight_only and not enable_token_export:
        raise ValueError("preflight-only batches require trainer.enable_token_export=true")
    return preflight_only


def has_weighted_sdpo_tokens(sdpo_weights: Tensor | None, loss_mask: Tensor | None = None) -> bool:
    if sdpo_weights is None:
        return False
    weighted = active_sdpo_weight_mask(sdpo_weights)
    if loss_mask is not None:
        weighted = weighted & loss_mask
    return bool(weighted.any())


def active_sdpo_weight_mask(sdpo_weights: Tensor) -> Tensor:
    if sdpo_weights.dtype == torch.bool or torch.is_complex(sdpo_weights):
        raise ValueError("sdpo_weights must contain finite non-negative numeric values")
    if not bool(torch.isfinite(sdpo_weights).all()):
        raise ValueError("sdpo_weights must contain finite weights")
    if bool((sdpo_weights < 0).any()):
        raise ValueError("sdpo_weights must contain non-negative weights")
    return sdpo_weights != 0


def _validate_sdpo_logits_and_temperatures(logits: Tensor, temperatures: Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError(f"SDPO logits must be rank 3 (batch, seq, vocab), got {tuple(logits.shape)}")
    if not torch.is_floating_point(logits):
        raise ValueError("SDPO logits must use a floating-point tensor dtype")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("SDPO logits must contain finite values")
    if temperatures.shape != logits.shape[:2]:
        raise ValueError(
            f"SDPO temperatures shape {tuple(temperatures.shape)} must match logits batch/seq {tuple(logits.shape[:2])}"
        )
    if not bool(torch.isfinite(temperatures).all()):
        raise ValueError("SDPO temperatures must contain finite values")
    if not bool((temperatures > 0).all()):
        raise ValueError("SDPO temperatures must be positive")


def _validate_sdpo_topk_token_ids(
    sdpo_topk_token_ids: Tensor,
    *,
    batch: int,
    seq: int,
    vocab: int,
    support_mask: Tensor | None = None,
) -> None:
    if sdpo_topk_token_ids.ndim != 3:
        raise ValueError(
            f"SDPO top-k token ids must be rank 3 (batch, seq, topk), got {tuple(sdpo_topk_token_ids.shape)}"
        )
    if sdpo_topk_token_ids.shape[:2] != (batch, seq):
        raise ValueError(
            "SDPO top-k token ids leading shape "
            f"{tuple(sdpo_topk_token_ids.shape[:2])} must match logits batch/seq {(batch, seq)}"
        )
    if sdpo_topk_token_ids.shape[-1] <= 0:
        raise ValueError("SDPO top-k token ids must have a non-empty top-k dimension")
    if (
        sdpo_topk_token_ids.dtype == torch.bool
        or torch.is_floating_point(sdpo_topk_token_ids)
        or torch.is_complex(sdpo_topk_token_ids)
    ):
        raise ValueError("SDPO top-k token ids must use an integer tensor dtype")
    if not bool(((sdpo_topk_token_ids >= 0) & (sdpo_topk_token_ids < vocab)).all()):
        raise ValueError(f"SDPO top-k token ids must be within vocabulary range [0, {vocab})")
    if support_mask is None:
        return
    if support_mask.shape != (batch, seq):
        raise ValueError(f"SDPO support mask shape {tuple(support_mask.shape)} must match batch/seq {(batch, seq)}")
    support_mask = support_mask.to(dtype=torch.bool, device=sdpo_topk_token_ids.device)
    if not bool(support_mask.any()):
        return
    sorted_ids = torch.sort(sdpo_topk_token_ids[support_mask], dim=-1).values
    if bool((sorted_ids[..., 1:] == sorted_ids[..., :-1]).any()):
        raise ValueError("SDPO top-k token ids must be distinct on supported token rows")


def _shift_logits_to_current_token(logits: Tensor) -> Tensor:
    batch, _, vocab = logits.shape
    left_pad_logit = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)
    return torch.cat([left_pad_logit, logits[:, :-1, :]], dim=1)


def gather_sdpo_student_topk_logprobs(
    logits: Tensor,
    temperatures: Tensor,
    sdpo_topk_token_ids: Tensor,
    support_mask: Tensor | None = None,
) -> Tensor:
    _validate_sdpo_logits_and_temperatures(logits, temperatures)
    _validate_sdpo_topk_token_ids(
        sdpo_topk_token_ids,
        batch=logits.shape[0],
        seq=logits.shape[1],
        vocab=logits.shape[2],
        support_mask=support_mask,
    )
    scaled_logits = logits / temperatures.unsqueeze(-1)
    shifted_log_probs = _shift_logits_to_current_token(scaled_logits).log_softmax(dim=-1)
    return torch.gather(shifted_log_probs, dim=-1, index=sdpo_topk_token_ids)


def select_sdpo_student_topk_support(
    logits: Tensor,
    temperatures: Tensor,
    topk: int,
) -> tuple[Tensor, Tensor]:
    """Select student top-k support from the trainer forward.

    The shifted convention matches ``gather_sdpo_student_topk_logprobs``:
    SDPO row ``i`` is selected from the previous next-token prediction, so row
    ``i`` describes the distribution used to predict token ``i``.
    """
    _validate_sdpo_logits_and_temperatures(logits, temperatures)
    if isinstance(topk, bool) or not isinstance(topk, int):
        raise ValueError("SDPO student top-k support requires integer topk")
    if topk <= 0:
        raise ValueError("SDPO student top-k support requires topk > 0")
    scaled_logits = logits / temperatures.unsqueeze(-1)
    shifted_logits = _shift_logits_to_current_token(scaled_logits)
    vocab_size = shifted_logits.shape[-1]
    if topk > vocab_size:
        raise ValueError(f"SDPO student top-k support requested topk={topk}, but vocab size is {vocab_size}")
    topk_logits, topk_token_ids = torch.topk(shifted_logits, topk, dim=-1)
    topk_log_probs = topk_logits - torch.logsumexp(shifted_logits, dim=-1, keepdim=True)
    return topk_token_ids, topk_log_probs


def should_export_sdpo_student_support(
    *,
    enable_token_export: bool,
    sdpo_weights: Tensor | None,
    loss_mask: Tensor | None,
    full_logit_distillation: bool,
    distillation_topk: int | None,
) -> bool:
    """Return whether the trainer should export student-selected top-k support.

    This is intentionally independent of ``preflight_only``: SDPO uses the
    same export shape for the preflight support pass and for final-batch smoke
    evidence after the orchestrator has hydrated teacher support.
    """
    return (
        enable_token_export
        and full_logit_distillation
        and distillation_topk is not None
        and has_weighted_sdpo_tokens(sdpo_weights, loss_mask)
    )


def require_sdpo_student_support_export_supported(*, cp_enabled: bool) -> None:
    if cp_enabled:
        raise NotImplementedError("SDPO student-support export is not supported with context parallelism yet")


def require_sdpo_student_support_logits(model_output: Mapping[str, Any]) -> Tensor:
    logits = model_output.get("logits")
    if logits is None:
        raise ValueError(
            "SDPO student-support export requires logits in the trainer output; "
            "set trainer.model.fused_lm_head_token_chunk_size='disabled'."
        )
    if not isinstance(logits, Tensor):
        raise ValueError("SDPO student-support export expected trainer output logits to be a tensor.")
    if not torch.is_floating_point(logits):
        raise ValueError("SDPO student-support export expected floating-point logits.")
    if logits.ndim != 3:
        raise ValueError(
            f"SDPO student-support export expected logits with shape (batch, seq, vocab), got {tuple(logits.shape)}."
        )
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("SDPO student-support export expected finite logits.")
    return logits
