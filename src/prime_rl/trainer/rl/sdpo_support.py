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
    if not bool(torch.isfinite(temperatures).all()):
        raise ValueError("SDPO temperatures must be finite")
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
    else:
        support_mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)

    result = logits.new_zeros(token_ids.shape)
    batch_positions = torch.nonzero(support_mask, as_tuple=False)
    if not len(batch_positions):
        return result

    batch_ids, positions = batch_positions.unbind(dim=1)
    first_token = positions == 0
    if bool(first_token.any()):
        result[batch_ids[first_token], positions[first_token]] = -torch.log(
            logits.new_tensor(float(logits.shape[-1]))
        )

    predicted = ~first_token
    if bool(predicted.any()):
        predicted_batches = batch_ids[predicted]
        predicted_positions = positions[predicted]
        source_positions = predicted_positions - 1
        selected_logits = logits[predicted_batches, source_positions]
        if not _all_finite_in_chunks(selected_logits):
            raise ValueError("SDPO logits and temperatures must be finite")
        scaled_logits = selected_logits / temperatures[predicted_batches, source_positions].unsqueeze(-1)
        selected_ids = token_ids[predicted_batches, predicted_positions]
        selected_logprobs = torch.gather(scaled_logits.log_softmax(dim=-1), dim=-1, index=selected_ids)
        result[predicted_batches, predicted_positions] = selected_logprobs
    return result


def select_sdpo_student_topk_support(logits: Tensor, temperatures: Tensor, topk: int) -> Tensor:
    if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
        raise ValueError("SDPO top-k support size must be a positive integer")
    if topk > logits.shape[-1]:
        raise ValueError(f"SDPO top-k support size {topk} exceeds vocabulary size {logits.shape[-1]}")
    _validate_logits_and_temperatures(logits, temperatures)
    # Positive temperature scaling preserves top-k membership. Select directly
    # from logits so large vocabularies do not require another full-size tensor.
    next_token_support = torch.topk(logits, topk, dim=-1).indices
    first_token_support = torch.topk(torch.zeros_like(logits[:, :1]), topk, dim=-1).indices
    return torch.cat([first_token_support, next_token_support[:, :-1]], dim=1)


def gather_sdpo_teacher_topk_logprobs(logits: Tensor, positions: Tensor, token_ids: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("SDPO teacher logits must have shape (1, seq, vocab)")
    if positions.ndim != 1 or token_ids.ndim != 2 or token_ids.shape[0] != positions.shape[0]:
        raise ValueError("SDPO teacher positions and support ids must align")
    if not bool(((positions >= 0) & (positions < logits.shape[1])).all()):
        raise ValueError("SDPO teacher positions must be within the teacher sequence")
    if not bool(((token_ids >= 0) & (token_ids < logits.shape[-1])).all()):
        raise ValueError("SDPO teacher support ids must be within the model vocabulary")
    result = logits.new_empty(token_ids.shape)
    first_token = positions == 0
    if bool(first_token.any()):
        result[first_token] = -torch.log(logits.new_tensor(float(logits.shape[-1])))
    predicted = ~first_token
    if bool(predicted.any()):
        selected_rows = logits[0, positions[predicted] - 1].log_softmax(dim=-1)
        result[predicted] = torch.gather(selected_rows, dim=-1, index=token_ids[predicted])
    return result


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
    if not torch.is_floating_point(logits) or not _all_finite_in_chunks(logits):
        raise ValueError("SDPO logits must contain finite floating-point values")
    if not bool(torch.isfinite(temperatures).all()) or not bool((temperatures > 0).all()):
        raise ValueError("SDPO temperatures must be finite and positive")


def _all_finite_in_chunks(tensor: Tensor, chunk_elements: int = 16 * 1024 * 1024) -> bool:
    flat = tensor.reshape(-1)
    return all(bool(torch.isfinite(chunk).all()) for chunk in flat.split(chunk_elements))
