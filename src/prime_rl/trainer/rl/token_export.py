import atexit
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from prime_rl.configs.trainer import DefaultLossConfig, TrainerConfig
from prime_rl.trainer.rl.loss import compute_importance_ratio_and_mismatch_kl

SCHEMA_VERSION = 1


class DisabledTokenExporter:
    def export(self, *args: Any, **kwargs: Any) -> None:
        return

    def mark_stable(self) -> None:
        return

    def close(self) -> None:
        return


class TokenExporter:
    def __init__(
        self,
        output_dir: Path,
        rank: int,
    ) -> None:
        self.rank = rank
        self.output_dir = output_dir / "token_exports"
        self._closed = False
        self._initialized_files: set[tuple[int, int]] = set()
        self._sequences_by_file: dict[tuple[int, int], int] = {}
        self._pending_stable_dirs: set[Path] = set()
        atexit.register(self.close)

    def export(
        self,
        step: int,
        micro_step: int,
        micro_batch: Mapping[str, Any],
        model_output: Mapping[str, Tensor],
        sequence_lengths: list[int],
        loss_config: Any,
        *,
        sdpo_topk_token_ids: Tensor | None = None,
        student_topk_logprobs: Tensor | None = None,
        teacher_topk_logprobs: Tensor | None = None,
        teacher_support_token_ids: Tensor | None = None,
        student_teacher_support_logprobs: Tensor | None = None,
        teacher_support_logprobs: Tensor | None = None,
    ) -> None:
        columns = _export_columns(micro_batch, model_output, loss_config)
        _check_lengths(columns)
        sdpo_support = _sparse_sdpo_support(
            micro_batch,
            sdpo_topk_token_ids=sdpo_topk_token_ids,
            student_topk_logprobs=student_topk_logprobs,
            teacher_topk_logprobs=teacher_topk_logprobs,
            teacher_support_token_ids=teacher_support_token_ids,
            student_teacher_support_logprobs=student_teacher_support_logprobs,
            teacher_support_logprobs=teacher_support_logprobs,
        )
        file_key = (step, self.rank)

        start = 0
        for micro_sequence_idx, length in enumerate(sequence_lengths):
            raw_end = start + length
            end = _trim_padding(columns, start, raw_end)
            if end > start and any(columns["loss_mask"][start:end]):
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "step": step,
                    "rank": self.rank,
                    "micro_step": micro_step,
                    "micro_sequence_idx": micro_sequence_idx,
                    "export_sequence_idx": self._sequences_by_file.get(file_key, 0),
                    "env_name": _first_non_empty(columns["env_names"][start:end]),
                    **_slice_columns(columns, start, end),
                }
                if sdpo_support is not None:
                    record["sdpo_support"] = _slice_sdpo_support(sdpo_support, start, end)
                    record["sdpo_teacher_replays"] = _slice_sdpo_teacher_replays(
                        micro_batch.get("sdpo_teacher_spans"),
                        start,
                        end,
                    )
                self._write(
                    record,
                    step,
                )
                self._sequences_by_file[file_key] = self._sequences_by_file.get(file_key, 0) + 1
            start = raw_end

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    def mark_stable(self) -> None:
        # The caller barriers first so a STABLE only lands after every rank flushed.
        while self._pending_stable_dirs:
            stable_dir = self._pending_stable_dirs.pop()
            (stable_dir / "STABLE").touch()

    def _export_dir(self, step: int) -> Path:
        return self.output_dir / f"step_{step}"

    def _export_file(self, step: int) -> Path:
        if self._closed:
            raise RuntimeError(f"Token exporter is closed for {self.output_dir}")

        step_dir = self._export_dir(step)
        try:
            step_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            if not step_dir.is_dir():
                raise
        return step_dir / f"rank_{self.rank}.jsonl"

    def _write(self, record: dict[str, Any], step: int) -> None:
        if self._closed:
            raise RuntimeError(f"Token exporter is closed for {self.output_dir}")

        file_key = (step, self.rank)
        mode = "a" if file_key in self._initialized_files else "w"
        export_file = self._export_file(step)
        with export_file.open(mode, encoding="utf-8") as file:
            file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._initialized_files.add(file_key)
        self._pending_stable_dirs.add(export_file.parent)


def setup_token_exporter(
    config: TrainerConfig, parallel_dims: Any, world: Any, logger: Any
) -> TokenExporter | DisabledTokenExporter:
    if not config.enable_token_export:
        return DisabledTokenExporter()
    if parallel_dims.cp_enabled and parallel_dims.world_mesh["cp"].get_local_rank() != 0:
        return DisabledTokenExporter()

    exporter = TokenExporter(config.output_dir, world.rank)
    logger.info(f"Writing token exports under {exporter.output_dir}")
    return exporter


def _export_columns(
    micro_batch: Mapping[str, Any], model_output: Mapping[str, Tensor], loss_config: Any
) -> dict[str, list[Any]]:
    token_ids = _tensor_to_ints(micro_batch["input_ids"])
    seq_len = len(token_ids)
    trainer_logprobs = model_output["logprobs"]
    export_tensors = _compute_export_tensors(micro_batch, trainer_logprobs, loss_config)

    return {
        "token_ids": token_ids,
        "position_ids": _tensor_to_ints(micro_batch["position_ids"]),
        "loss_mask": _tensor_to_bools(micro_batch["loss_mask"]),
        "advantages": _tensor_to_floats(micro_batch["advantages"]),
        "inference_logprobs": _tensor_to_floats(micro_batch["inference_logprobs"]),
        "trainer_logprobs": _tensor_to_floats(trainer_logprobs),
        "entropy": _tensor_to_floats(model_output["entropy"]),
        "mismatch_kl": _optional_tensor_to_floats(export_tensors["mismatch_kl"], seq_len),
        "log_importance_ratio": _optional_tensor_to_floats(export_tensors["log_importance_ratio"], seq_len),
        "importance_ratio": _optional_tensor_to_floats(export_tensors["importance_ratio"], seq_len),
        "prob_delta": _optional_tensor_to_floats(export_tensors["prob_delta"], seq_len),
        "is_masked": _optional_tensor_to_bools(export_tensors["is_masked"], seq_len),
        "is_masked_high": _optional_tensor_to_bools(export_tensors["is_masked_high"], seq_len),
        "is_masked_low": _optional_tensor_to_bools(export_tensors["is_masked_low"], seq_len),
        # Component weight streams; ``None`` columns mean the defaults (rl 1.0
        # on the loss mask, no ce/ref_kl/sdpo component).
        "rl_weights": _optional_tensor_to_floats(micro_batch.get("rl_weights"), seq_len),
        "ce_weights": _optional_tensor_to_floats(micro_batch.get("ce_weights"), seq_len),
        "ref_kl_weights": _optional_tensor_to_floats(micro_batch.get("ref_kl_weights"), seq_len),
        "sdpo_weights": _optional_tensor_to_floats(micro_batch.get("sdpo_weights"), seq_len),
        "env_names": list(micro_batch["env_names"]),
    }


def _compute_export_tensors(
    micro_batch: Mapping[str, Any], trainer_logprobs: Tensor, loss_config: Any
) -> dict[str, Tensor | None]:
    fields: dict[str, Tensor | None] = {
        "log_importance_ratio": None,
        "importance_ratio": None,
        "mismatch_kl": None,
        "prob_delta": None,
        "is_masked": None,
        "is_masked_high": None,
        "is_masked_low": None,
    }
    # Ratio-based fields are meaningless when no token has sampling logprobs
    # (e.g. pure CE batches distilling frozen-model tokens): no rl member
    # (stream present but all-zero) and no ref_kl member.
    rl_weights = micro_batch.get("rl_weights")
    ref_kl_weights = micro_batch.get("ref_kl_weights")
    sdpo_weights = micro_batch.get("sdpo_weights")
    no_rl = rl_weights is not None and not bool((rl_weights != 0).any())
    no_ref_kl = ref_kl_weights is None or not bool((ref_kl_weights != 0).any())
    no_sdpo = sdpo_weights is None or not bool((sdpo_weights != 0).any())
    if no_rl and no_ref_kl and no_sdpo:
        return fields

    inference_logprobs = micro_batch["inference_logprobs"].to(trainer_logprobs.device)
    loss_mask = micro_batch["loss_mask"].to(trainer_logprobs.device)
    advantages = micro_batch["advantages"].to(trainer_logprobs.device)
    with torch.no_grad():
        log_ratio, ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(trainer_logprobs, inference_logprobs)
        prob_delta = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
        fields["log_importance_ratio"] = log_ratio
        fields["importance_ratio"] = ratio
        fields["mismatch_kl"] = mismatch_kl
        fields["prob_delta"] = prob_delta
        if isinstance(loss_config, DefaultLossConfig):
            invalid_high = prob_delta > loss_config.dppo_mask_high
            invalid_low = prob_delta < -loss_config.dppo_mask_low
            positive_advantages = advantages > 0
            negative_advantages = advantages < 0
            invalid = torch.where(positive_advantages, invalid_high, invalid_low)
            fields["is_masked"] = loss_mask & invalid
            fields["is_masked_high"] = loss_mask & positive_advantages & invalid_high
            fields["is_masked_low"] = loss_mask & negative_advantages & invalid_low
    return fields


def _sparse_sdpo_support(
    micro_batch: Mapping[str, Any],
    *,
    sdpo_topk_token_ids: Tensor | None,
    student_topk_logprobs: Tensor | None,
    teacher_topk_logprobs: Tensor | None,
    teacher_support_token_ids: Tensor | None,
    student_teacher_support_logprobs: Tensor | None,
    teacher_support_logprobs: Tensor | None,
) -> list[dict[str, Any]] | None:
    tensors = (
        sdpo_topk_token_ids,
        student_topk_logprobs,
        teacher_topk_logprobs,
        teacher_support_token_ids,
        student_teacher_support_logprobs,
        teacher_support_logprobs,
    )
    if all(tensor is None for tensor in tensors):
        return None
    if any(tensor is None for tensor in tensors):
        raise ValueError("SDPO support export requires token ids and both student and teacher logprobs")

    assert sdpo_topk_token_ids is not None
    assert student_topk_logprobs is not None
    assert teacher_topk_logprobs is not None
    assert teacher_support_token_ids is not None
    assert student_teacher_support_logprobs is not None
    assert teacher_support_logprobs is not None
    seq_len = micro_batch["input_ids"].numel()
    expected_prefix = (1, seq_len)
    if sdpo_topk_token_ids.ndim != 3 or sdpo_topk_token_ids.shape[:2] != expected_prefix:
        raise ValueError("SDPO support token ids must have shape (1, seq, topk)")
    if student_topk_logprobs.shape != sdpo_topk_token_ids.shape:
        raise ValueError("SDPO student top-k logprobs must align with support token ids")
    if teacher_topk_logprobs.shape != sdpo_topk_token_ids.shape:
        raise ValueError("SDPO teacher top-k logprobs must align with support token ids")
    for name, tensor in (
        ("teacher support token ids", teacher_support_token_ids),
        ("student logprobs on teacher support", student_teacher_support_logprobs),
        ("teacher support logprobs", teacher_support_logprobs),
    ):
        if tensor.shape != sdpo_topk_token_ids.shape:
            raise ValueError(f"SDPO {name} must align with student support token ids")

    weights = micro_batch.get("sdpo_weights")
    if weights is None:
        if micro_batch.get("sdpo_teacher_spans"):
            raise ValueError("SDPO teacher spans require sdpo_weights")
        # Every rank participates in global SDPO teacher collectives. A rank
        # without a local SDPO sample therefore receives support tensors but
        # has no sparse positions to export.
        return []
    active = micro_batch["loss_mask"].reshape(-1) & (weights.reshape(-1) != 0)
    positions = torch.nonzero(active, as_tuple=False).flatten()
    if not len(positions):
        return []

    selected_ids = sdpo_topk_token_ids[0, positions]
    selected_student = student_topk_logprobs[0, positions]
    selected_teacher = teacher_topk_logprobs[0, positions]
    selected_teacher_ids = teacher_support_token_ids[0, positions]
    selected_student_on_teacher = student_teacher_support_logprobs[0, positions]
    selected_teacher_support = teacher_support_logprobs[0, positions]
    if not all(
        bool(torch.isfinite(values).all())
        for values in (selected_student, selected_teacher, selected_student_on_teacher, selected_teacher_support)
    ):
        raise ValueError("SDPO support export logprobs must be finite on active tokens")

    position_values = _tensor_to_ints(positions)
    token_id_values = selected_ids.detach().to(device="cpu").tolist()
    student_values = selected_student.detach().to(dtype=torch.float32, device="cpu").tolist()
    teacher_values = selected_teacher.detach().to(dtype=torch.float32, device="cpu").tolist()
    teacher_id_values = selected_teacher_ids.detach().to(device="cpu").tolist()
    student_on_teacher_values = (
        selected_student_on_teacher.detach().to(dtype=torch.float32, device="cpu").tolist()
    )
    teacher_support_values = selected_teacher_support.detach().to(dtype=torch.float32, device="cpu").tolist()
    return [
        {
            "position": position,
            "student_support": {
                "token_ids": [int(token_id) for token_id in token_ids],
                "student_logprobs": [_json_float(value) for value in student_logprobs],
                "teacher_logprobs": [_json_float(value) for value in teacher_logprobs],
            },
            "teacher_support": {
                "token_ids": [int(token_id) for token_id in teacher_ids],
                "student_logprobs": [_json_float(value) for value in student_on_teacher],
                "teacher_logprobs": [_json_float(value) for value in teacher_support],
            },
        }
        for (
            position,
            token_ids,
            student_logprobs,
            teacher_logprobs,
            teacher_ids,
            student_on_teacher,
            teacher_support,
        ) in zip(
            position_values,
            token_id_values,
            student_values,
            teacher_values,
            teacher_id_values,
            student_on_teacher_values,
            teacher_support_values,
            strict=True,
        )
    ]


def _slice_sdpo_support(
    support: list[dict[str, Any]], start: int, end: int
) -> list[dict[str, Any]]:
    return [
        {**entry, "position": entry["position"] - start}
        for entry in support
        if start <= entry["position"] < end
    ]


def _slice_sdpo_teacher_replays(spans: Sequence[Any] | None, start: int, end: int) -> list[dict[str, Any]]:
    result = []
    for span in spans or []:
        positions = list(span.student_positions)
        if not positions:
            continue
        selected = [start <= position < end for position in positions]
        if any(selected) and not all(selected):
            raise ValueError("one SDPO teacher replay cannot span packed sequences")
        if not any(selected):
            continue
        result.append(
            {
                "prefix_ids": list(span.prefix_ids),
                "completion_ids": list(span.completion_ids),
                "student_positions": [position - start for position in positions],
                "target_offsets": list(span.target_offsets),
            }
        )
    return result


def _tensor_to_ints(tensor: Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_to_bools(tensor: Tensor) -> list[bool]:
    return [bool(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_to_floats(tensor: Tensor) -> list[float | None]:
    values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
    return [_json_float(value) for value in values]


def _optional_tensor_to_floats(tensor: Tensor | None, seq_len: int) -> list[float | None]:
    if tensor is None:
        return [None] * seq_len
    return _tensor_to_floats(tensor)


def _optional_tensor_to_bools(tensor: Tensor | None, seq_len: int) -> list[bool | None]:
    if tensor is None:
        return [None] * seq_len
    return _tensor_to_bools(tensor)


def _check_lengths(columns: Mapping[str, Sequence[Any]]) -> None:
    lengths = {key: len(values) for key, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Token export fields must have aligned lengths, got {lengths}")


def _slice_columns(columns: Mapping[str, Sequence[Any]], start: int, end: int) -> dict[str, list[Any]]:
    return {key: list(values[start:end]) for key, values in columns.items() if key != "env_names"}


def _trim_padding(columns: Mapping[str, Sequence[Any]], start: int, end: int) -> int:
    env_names = columns["env_names"]
    loss_mask = columns["loss_mask"]
    while end > start and env_names[end - 1] == "" and not loss_mask[end - 1]:
        end -= 1
    return end


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _first_non_empty(values: Sequence[str]) -> str | None:
    for value in values:
        if value:
            return value
    return None
