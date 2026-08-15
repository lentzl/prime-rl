from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.transport import SDPOTeacherSpan


def active_sdpo_weight_mask(weights: Tensor) -> Tensor:
    if weights.dtype == torch.bool or torch.is_complex(weights):
        raise ValueError("sdpo_weights must contain finite non-negative numeric values")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("sdpo_weights must contain finite values")
    if bool((weights < 0).any()):
        raise ValueError("sdpo_weights must contain non-negative values")
    return weights != 0


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
    left_pad = torch.zeros_like(scaled_logits[:, :1])
    current_token_logprobs = torch.cat([left_pad, scaled_logits[:, :-1]], dim=1).log_softmax(dim=-1)
    return torch.gather(current_token_logprobs, dim=-1, index=token_ids)


def select_sdpo_student_topk_support(logits: Tensor, temperatures: Tensor, topk: int) -> Tensor:
    if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
        raise ValueError("SDPO top-k support size must be a positive integer")
    if topk > logits.shape[-1]:
        raise ValueError(f"SDPO top-k support size {topk} exceeds vocabulary size {logits.shape[-1]}")
    _validate_logits_and_temperatures(logits, temperatures)
    scaled_logits = logits / temperatures.unsqueeze(-1)
    left_pad = torch.zeros_like(scaled_logits[:, :1])
    current_token_logits = torch.cat([left_pad, scaled_logits[:, :-1]], dim=1)
    return torch.topk(current_token_logits, topk, dim=-1).indices


def gather_sdpo_teacher_topk_logprobs(logits: Tensor, positions: Tensor, token_ids: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("SDPO teacher logits must have shape (1, seq, vocab)")
    if positions.ndim != 1 or token_ids.ndim != 2 or token_ids.shape[0] != positions.shape[0]:
        raise ValueError("SDPO teacher positions and support ids must align")
    if not bool(((positions >= 0) & (positions < logits.shape[1])).all()):
        raise ValueError("SDPO teacher positions must be within the teacher sequence")
    if not bool(((token_ids >= 0) & (token_ids < logits.shape[-1])).all()):
        raise ValueError("SDPO teacher support ids must be within the model vocabulary")
    left_pad = torch.zeros_like(logits[:, :1])
    current_token_logprobs = torch.cat([left_pad, logits[:, :-1]], dim=1).log_softmax(dim=-1)
    selected_rows = current_token_logprobs[0, positions]
    return torch.gather(selected_rows, dim=-1, index=token_ids)


def pack_sdpo_teacher_spans(
    spans: list[SDPOTeacherSpan] | None,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    input_ids: list[int] = []
    position_ids: list[int] = []
    seq_lens: list[int] = []
    teacher_positions: list[int] = []
    student_positions: list[int] = []
    for span in spans or []:
        sequence = span.prefix_ids + span.completion_ids
        sequence_start = len(input_ids)
        input_ids.extend(sequence)
        position_ids.extend(range(len(sequence)))
        seq_lens.append(len(sequence))
        teacher_positions.extend(sequence_start + len(span.prefix_ids) + offset for offset in span.target_offsets)
        student_positions.extend(span.student_positions)
    if not input_ids:
        input_ids = [1]
        position_ids = [0]
        seq_lens = [1]
    return input_ids, position_ids, seq_lens, teacher_positions, student_positions


def pack_sdpo_teacher_span_batches(
    spans: list[SDPOTeacherSpan] | None,
    max_seq_len: int,
) -> list[tuple[list[int], list[int], list[int], list[int], list[int]]]:
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


def _validate_logits_and_temperatures(logits: Tensor, temperatures: Tensor) -> None:
    if logits.ndim != 3 or temperatures.shape != logits.shape[:2]:
        raise ValueError("SDPO logits and temperatures must align as (batch, seq, vocab) and (batch, seq)")
    if not torch.is_floating_point(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError("SDPO logits must contain finite floating-point values")
    if not bool(torch.isfinite(temperatures).all()) or not bool((temperatures > 0).all()):
        raise ValueError("SDPO temperatures must be finite and positive")
