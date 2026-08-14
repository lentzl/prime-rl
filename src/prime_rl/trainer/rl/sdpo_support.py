from __future__ import annotations

import math

import torch
from torch import Tensor

from prime_rl.transport import SDPOTeacherSpan

SDPOTeacherBatch = tuple[list[int], list[int], list[int], list[int], list[int]]


def active_sdpo_weight_mask(weights: Tensor) -> Tensor:
    if weights.dtype == torch.bool or torch.is_complex(weights):
        raise ValueError("sdpo_weights must contain finite non-negative numeric values")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("sdpo_weights must contain finite values")
    if bool((weights < 0).any()):
        raise ValueError("sdpo_weights must contain non-negative values")
    return weights != 0


def align_sdpo_predictor_support_to_current_tokens(predictor_token_ids: Tensor, vocab_size: int) -> Tensor:
    """Shift predictor-row support to the current-token convention used by RL batches."""
    if predictor_token_ids.ndim != 3 or predictor_token_ids.shape[1] == 0 or predictor_token_ids.shape[2] == 0:
        raise ValueError("SDPO predictor support must have shape (batch, nonempty seq, nonempty topk)")
    if predictor_token_ids.dtype != torch.long:
        raise ValueError("SDPO predictor support ids must use torch.long dtype")
    if not bool(((predictor_token_ids >= 0) & (predictor_token_ids < vocab_size)).all()):
        raise ValueError("SDPO predictor support ids must be within the model vocabulary")

    topk = predictor_token_ids.shape[-1]
    first_support = torch.arange(topk, device=predictor_token_ids.device).view(1, 1, topk)
    first_support = first_support.expand(predictor_token_ids.shape[0], 1, topk)
    return torch.cat([first_support, predictor_token_ids[:, :-1]], dim=1)


def align_sdpo_predictor_logprobs_to_current_tokens(predictor_logprobs: Tensor, vocab_size: int) -> Tensor:
    """Shift predictor-row selected log-probabilities to current-token positions."""
    if predictor_logprobs.ndim != 3 or predictor_logprobs.shape[1] == 0 or predictor_logprobs.shape[2] == 0:
        raise ValueError("SDPO predictor log-probabilities must have shape (batch, nonempty seq, nonempty topk)")
    uniform = torch.full(
        (predictor_logprobs.shape[0], 1, predictor_logprobs.shape[2]),
        -math.log(vocab_size),
        dtype=predictor_logprobs.dtype,
        device=predictor_logprobs.device,
    )
    return torch.cat([uniform, predictor_logprobs[:, :-1]], dim=1)


def place_sdpo_teacher_support_on_predictors(
    positions: Tensor,
    token_ids: Tensor,
    teacher_seq_len: int,
) -> Tensor:
    """Place current-token support ids on the teacher rows that predict them."""
    if positions.ndim != 1 or token_ids.ndim != 2 or token_ids.shape[0] != positions.shape[0]:
        raise ValueError("SDPO teacher positions and support ids must align as (targets) and (targets, topk)")
    if teacher_seq_len <= 0 or not bool(((positions >= 0) & (positions < teacher_seq_len)).all()):
        raise ValueError("SDPO teacher positions must be within the teacher sequence")
    positive_positions = positions[positions > 0]
    if positive_positions.numel() != torch.unique(positive_positions).numel():
        raise ValueError("SDPO teacher positions must be unique")

    support = torch.zeros((1, teacher_seq_len, token_ids.shape[-1]), dtype=token_ids.dtype, device=token_ids.device)
    positive = positions > 0
    support[0, positions[positive] - 1] = token_ids[positive]
    return support


def gather_sdpo_teacher_fused_topk_logprobs(
    predictor_logprobs: Tensor,
    positions: Tensor,
    vocab_size: int,
) -> Tensor:
    """Gather fused teacher scores at current-token target positions."""
    if predictor_logprobs.ndim != 3 or predictor_logprobs.shape[0] != 1:
        raise ValueError("SDPO teacher predictor log-probabilities must have shape (1, seq, topk)")
    if positions.ndim != 1 or not bool(((positions >= 0) & (positions < predictor_logprobs.shape[1])).all()):
        raise ValueError("SDPO teacher positions must be within the teacher sequence")
    scores = torch.full(
        (positions.shape[0], predictor_logprobs.shape[-1]),
        -math.log(vocab_size),
        dtype=predictor_logprobs.dtype,
        device=predictor_logprobs.device,
    )
    positive = positions > 0
    scores[positive] = predictor_logprobs[0, positions[positive] - 1]
    return scores


def gather_sdpo_student_topk_logprobs(
    logits: Tensor,
    temperatures: Tensor,
    token_ids: Tensor,
    *,
    support_mask: Tensor | None = None,
) -> Tensor:
    if logits.ndim != 3:
        raise ValueError(f"SDPO logits must have shape (batch, seq, vocab), got {tuple(logits.shape)}")
    if temperatures.shape != logits.shape[:2]:
        raise ValueError("SDPO temperatures must align with logits")
    if token_ids.ndim != 3 or token_ids.shape[:2] != logits.shape[:2]:
        raise ValueError("SDPO top-k token ids must have shape (batch, seq, topk)")
    if token_ids.dtype == torch.bool or torch.is_floating_point(token_ids) or torch.is_complex(token_ids):
        raise ValueError("SDPO top-k token ids must use an integer dtype")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(temperatures).all()):
        raise ValueError("SDPO logits and temperatures must be finite")
    if not bool((temperatures > 0).all()):
        raise ValueError("SDPO temperatures must be positive")
    if not bool(((token_ids >= 0) & (token_ids < logits.shape[-1])).all()):
        raise ValueError("SDPO top-k token ids must be within the model vocabulary")
    if support_mask is not None:
        if support_mask.shape != logits.shape[:2]:
            raise ValueError("SDPO support mask must align with logits")
        selected = token_ids[support_mask]
        sorted_ids = selected.sort(dim=-1).values
        if selected.numel() and bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
            raise ValueError("SDPO top-k token ids must be distinct on supported rows")

    scaled_logits = logits / temperatures.unsqueeze(-1)
    uniform = torch.full(
        token_ids[:, :1].shape,
        -math.log(logits.shape[-1]),
        dtype=logits.dtype,
        device=logits.device,
    )
    predictor_logits = scaled_logits[:, :-1]
    selected_logits = torch.gather(predictor_logits, dim=-1, index=token_ids[:, 1:])
    selected_logprobs = selected_logits - torch.logsumexp(predictor_logits, dim=-1, keepdim=True)
    return torch.cat([uniform, selected_logprobs], dim=1)


def select_sdpo_student_topk_support(logits: Tensor, temperatures: Tensor, topk: int) -> Tensor:
    if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
        raise ValueError("SDPO top-k support size must be a positive integer")
    if topk > logits.shape[-1]:
        raise ValueError(f"SDPO top-k support size {topk} exceeds vocabulary size {logits.shape[-1]}")
    _validate_logits_and_temperatures(logits, temperatures)
    scaled_logits = logits / temperatures.unsqueeze(-1)
    first_support = torch.arange(topk, device=logits.device).view(1, 1, topk).expand(logits.shape[0], 1, topk)
    predictor_support = torch.topk(scaled_logits[:, :-1], topk, dim=-1).indices
    return torch.cat([first_support, predictor_support], dim=1)


def gather_sdpo_teacher_topk_logprobs(logits: Tensor, positions: Tensor, token_ids: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("SDPO teacher logits must have shape (1, seq, vocab)")
    if positions.ndim != 1 or token_ids.ndim != 2 or token_ids.shape[0] != positions.shape[0]:
        raise ValueError("SDPO teacher positions and support ids must align")
    if not bool(((positions >= 0) & (positions < logits.shape[1])).all()):
        raise ValueError("SDPO teacher positions must be within the teacher sequence")
    if not bool(((token_ids >= 0) & (token_ids < logits.shape[-1])).all()):
        raise ValueError("SDPO teacher support ids must be within the model vocabulary")
    predictor_positions = (positions - 1).clamp_min(0)
    predictor_logits = logits[0, predictor_positions]
    selected_logits = torch.gather(predictor_logits, dim=-1, index=token_ids)
    selected_logprobs = selected_logits - torch.logsumexp(predictor_logits, dim=-1, keepdim=True)
    uniform = torch.full_like(selected_logprobs, -math.log(logits.shape[-1]))
    return torch.where((positions == 0).unsqueeze(-1), uniform, selected_logprobs)


def pack_sdpo_teacher_spans(
    spans: list[SDPOTeacherSpan] | None,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    input_ids: list[int] = []
    position_ids: list[int] = []
    teacher_positions: list[int] = []
    student_positions: list[int] = []
    sequence_lengths: list[int] = []
    for span in spans or []:
        sequence = span.prefix_ids + span.completion_ids
        sequence_start = len(input_ids)
        input_ids.extend(sequence)
        position_ids.extend(range(len(sequence)))
        teacher_positions.extend(sequence_start + len(span.prefix_ids) + offset for offset in span.target_offsets)
        student_positions.extend(span.student_positions)
        sequence_lengths.append(len(sequence))
    if not input_ids:
        input_ids = [1]
        position_ids = [0]
        sequence_lengths = [1]
    return input_ids, position_ids, teacher_positions, student_positions, sequence_lengths


def pack_sdpo_teacher_span_batches(
    spans: list[SDPOTeacherSpan] | None,
    max_seq_len: int,
) -> list[SDPOTeacherBatch]:
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("SDPO teacher batch length must be a positive integer")

    batches: list[tuple[list[int], list[int], list[int], list[int], list[int]]] = []
    current: list[SDPOTeacherSpan] = []
    current_len = 0
    for span in spans or []:
        span_len = len(span.prefix_ids) + len(span.completion_ids)
        if span_len > max_seq_len:
            raise ValueError(f"SDPO teacher span has {span_len} tokens, exceeding model.seq_len={max_seq_len}")
        if current and current_len + span_len > max_seq_len:
            batches.append(pack_sdpo_teacher_spans(current))
            current = []
            current_len = 0
        current.append(span)
        current_len += span_len
    if current:
        batches.append(pack_sdpo_teacher_spans(current))
    return batches


def pad_sdpo_teacher_batches(
    batches: list[SDPOTeacherBatch],
    target_count: int,
    *,
    dummy_token_id: int,
) -> list[SDPOTeacherBatch]:
    """Pad rank-local replay so every FSDP rank runs the same teacher forwards."""
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < len(batches):
        raise ValueError("SDPO teacher target batch count must cover all local batches")
    dummy = ([dummy_token_id], [0], [], [], [1])
    return [*batches, *([dummy] * (target_count - len(batches)))]


def _validate_logits_and_temperatures(logits: Tensor, temperatures: Tensor) -> None:
    if logits.ndim != 3 or temperatures.shape != logits.shape[:2]:
        raise ValueError("SDPO logits and temperatures must align as (batch, seq, vocab) and (batch, seq)")
    if not torch.is_floating_point(logits):
        raise ValueError("SDPO logits must contain finite floating-point values")
    logit_min, logit_max = torch.aminmax(logits)
    if not bool(torch.isfinite(logit_min) & torch.isfinite(logit_max)):
        raise ValueError("SDPO logits must contain finite floating-point values")
    if not bool(torch.isfinite(temperatures).all()) or not bool((temperatures > 0).all()):
        raise ValueError("SDPO temperatures must be finite and positive")
