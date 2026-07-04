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
from prime_rl.trainer.rl.sdpo_loss import compute_rollout_is_weights
from prime_rl.trainer.rl.sdpo_train_support import active_sdpo_weight_mask
from prime_rl.transport.sdpo import has_active_sdpo_weights, is_active_sdpo_weight

SCHEMA_VERSION = 2


class DisabledTokenExporter:
    def export(self, *args: Any, **kwargs: Any) -> None:
        return

    def mark_stable(self, ready_run_ids: set[str] | None = None) -> None:
        return

    def close(self) -> None:
        return


def token_export_ready_run_ids(
    micro_batches: Sequence[Mapping[str, Any]],
    *,
    preflight_only: bool,
    multi_run_manager: Any,
) -> set[str]:
    if preflight_only:
        preflight_complete_by_run: dict[str, bool] = {}
        for micro_batch in micro_batches:
            run_id = micro_batch["run_id"]
            if run_id is None:
                continue
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("preflight token exports require run_id to be a non-empty string when present")
            preflight_step_complete = micro_batch.get("preflight_step_complete")
            if not isinstance(preflight_step_complete, bool):
                raise ValueError("preflight_step_complete must be a boolean on preflight micro batches")
            previous = preflight_complete_by_run.get(run_id)
            if previous is not None and previous != preflight_step_complete:
                raise ValueError(f"preflight_step_complete disagrees across micro batches for run_id {run_id!r}")
            preflight_complete_by_run[run_id] = preflight_step_complete
        ready_run_ids: set[str] = set()
        for run_id, preflight_step_complete in preflight_complete_by_run.items():
            if preflight_step_complete:
                ready_run_ids.add(run_id)
        return ready_run_ids
    return {
        multi_run_manager.idx_2_id[idx]
        for idx in multi_run_manager.ready_to_update_idxs
        if idx in multi_run_manager.idx_2_id
    }


class TokenExporter:
    def __init__(
        self,
        output_dir: Path,
        rank: int,
    ) -> None:
        self.rank = rank
        self.output_dir = output_dir / "token_exports"
        self._closed = False
        self._initialized_files: set[tuple[str | None, int, int]] = set()
        self._sequences_by_file: dict[tuple[str | None, int, int], int] = {}
        self._pending_stable_dirs: dict[str | None, set[Path]] = {}
        atexit.register(self.close)

    def export(
        self,
        step: int,
        micro_step: int,
        micro_batch: Mapping[str, Any],
        model_output: Mapping[str, Tensor],
        sequence_lengths: list[int],
        loss_config: Any,
        sdpo_loss_config: Any | None = None,
    ) -> None:
        columns = _export_columns(
            micro_batch,
            model_output,
            loss_config,
            sequence_lengths=sequence_lengths,
            sdpo_loss_config=sdpo_loss_config,
        )
        _check_lengths(columns)
        _check_sequence_lengths(sequence_lengths, len(columns["token_ids"]))
        run_id = micro_batch.get("run_id")
        export_step = micro_batch.get("run_step") if micro_batch.get("run_step") is not None else step
        preflight_only = _preflight_only_flag(micro_batch)
        sample_ids = micro_batch.get("sample_ids")
        if sample_ids is not None and len(sample_ids) != len(sequence_lengths):
            raise ValueError(f"sample_ids length {len(sample_ids)} != sequence_lengths length {len(sequence_lengths)}")
        if sample_ids is not None:
            for idx, sample_id in enumerate(sample_ids):
                if sample_id is not None and not _is_non_blank_string(sample_id):
                    raise ValueError(f"sample_ids[{idx}] must be null or a non-empty string")
        file_key = (run_id, export_step, self.rank)
        self._reset_missing_file_state(file_key, run_id, export_step)

        start = 0
        for micro_sequence_idx, length in enumerate(sequence_lengths):
            raw_end = start + length
            end = _trim_padding(columns, start, raw_end)
            if end > start and any(columns["loss_mask"][start:end]):
                sample_id = sample_ids[micro_sequence_idx] if sample_ids is not None else None
                has_active_sdpo = _has_nonzero_sdpo_weight(columns["sdpo_weights"][start:end])
                env_name = _first_non_empty(columns["env_names"][start:end])
                if has_active_sdpo:
                    if not _is_non_blank_string(sample_id):
                        raise ValueError(
                            "SDPO token export records require a non-empty sample_id "
                            f"(micro_sequence_idx={micro_sequence_idx}, export_step={export_step})"
                        )
                    if not _is_non_blank_string(env_name):
                        raise ValueError(
                            "SDPO token export records require a non-empty env_name "
                            f"(micro_sequence_idx={micro_sequence_idx}, export_step={export_step})"
                        )
                if preflight_only and _has_supported_sdpo_rows(
                    columns["sdpo_weights"][start:end],
                    columns["sdpo_topk_token_ids"][start:end],
                    columns["sdpo_topk_logprobs"][start:end],
                ):
                    raise ValueError(
                        "preflight-only SDPO token exports must not carry transported teacher top-k support"
                    )
                if not preflight_only and _has_missing_or_placeholder_sdpo_rows(
                    columns["sdpo_weights"][start:end],
                    columns["sdpo_topk_token_ids"][start:end],
                    columns["sdpo_topk_logprobs"][start:end],
                ):
                    raise ValueError(
                        "final SDPO token exports require transported teacher top-k support at every weighted token"
                    )
                self._write(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "step": step,
                        "export_step": export_step,
                        "rank": self.rank,
                        "micro_step": micro_step,
                        "micro_sequence_idx": micro_sequence_idx,
                        "export_sequence_idx": self._sequences_by_file.get(file_key, 0),
                        "preflight_only": preflight_only,
                        "sample_id": sample_id,
                        "run_id": run_id,
                        "env_name": env_name,
                        **_slice_columns(columns, start, end),
                    },
                    run_id,
                    export_step,
                )
                self._sequences_by_file[file_key] = self._sequences_by_file.get(file_key, 0) + 1
            start = raw_end

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    def mark_stable(self, ready_run_ids: set[str] | None = None) -> None:
        # A single run step's sequences can span multiple trainer steps (the packer
        # splits a run step across packs when it exceeds the token budget). Only
        # finalize a run's export dir once that run step is fully consumed — the same
        # `ready_to_update` signal that gates its optimizer step. ``run_id is None``
        # is single-run export, where a step never spans trainer steps, so mark it now.
        # The caller barriers first so a STABLE only lands after every rank flushed.
        ready_run_ids = ready_run_ids or set()
        for run_id in [rid for rid in self._pending_stable_dirs if rid is None or rid in ready_run_ids]:
            for stable_dir in self._pending_stable_dirs.pop(run_id):
                (stable_dir / "STABLE").touch()

    def _export_dir(self, export_step: int, run_id: str | None) -> Path:
        if run_id is not None:
            return self.output_dir.parent / run_id / "token_exports" / f"step_{export_step}"
        return self.output_dir / f"step_{export_step}"

    def _export_file(self, export_step: int, run_id: str | None) -> Path:
        if self._closed:
            raise RuntimeError(f"Token exporter is closed for {self.output_dir}")

        step_dir = self._export_dir(export_step, run_id)
        try:
            step_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            if not step_dir.is_dir():
                raise
        return step_dir / f"rank_{self.rank}.jsonl"

    def _write(self, record: dict[str, Any], run_id: str | None, export_step: int) -> None:
        if self._closed:
            raise RuntimeError(f"Token exporter is closed for {self.output_dir}")

        file_key = (run_id, export_step, self.rank)
        mode = "a" if file_key in self._initialized_files else "w"
        export_file = self._export_file(export_step, run_id)
        stable_marker = export_file.parent / "STABLE"
        if stable_marker.exists():
            stable_marker.unlink()
        with export_file.open(mode, encoding="utf-8") as file:
            file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._initialized_files.add(file_key)
        self._pending_stable_dirs.setdefault(run_id, set()).add(export_file.parent)

    def _reset_missing_file_state(
        self, file_key: tuple[str | None, int, int], run_id: str | None, export_step: int
    ) -> None:
        if file_key not in self._initialized_files:
            return
        export_file = self._export_dir(export_step, run_id) / f"rank_{self.rank}.jsonl"
        if export_file.exists():
            return
        self._initialized_files.discard(file_key)
        self._sequences_by_file.pop(file_key, None)
        pending_dirs = self._pending_stable_dirs.get(run_id)
        if pending_dirs is not None:
            pending_dirs.discard(export_file.parent)
            if not pending_dirs:
                self._pending_stable_dirs.pop(run_id, None)


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
    micro_batch: Mapping[str, Any],
    model_output: Mapping[str, Tensor],
    loss_config: Any,
    *,
    sequence_lengths: Sequence[int],
    sdpo_loss_config: Any | None = None,
) -> dict[str, list[Any]]:
    token_ids = _tensor_to_non_negative_ints(micro_batch["input_ids"], "input_ids")
    seq_len = len(token_ids)
    trainer_logprobs = model_output["logprobs"]
    export_tensors = _compute_export_tensors(micro_batch, trainer_logprobs, loss_config)
    sdpo_weights = _optional_weight_tensor_to_floats(micro_batch.get("sdpo_weights"), seq_len, "sdpo_weights")
    loss_mask = _tensor_to_bools(micro_batch["loss_mask"], "loss_mask")
    _validate_sdpo_weights_follow_loss_mask(sdpo_weights, loss_mask)
    sdpo_topk_token_ids = _optional_tensor_rows_to_ints(
        micro_batch.get("sdpo_topk_token_ids"), seq_len, "sdpo_topk_token_ids"
    )
    sdpo_topk_logprobs = _optional_tensor_rows_to_logprobs(
        micro_batch.get("sdpo_topk_logprobs"), seq_len, "sdpo_topk_logprobs"
    )
    sdpo_rollout_is_weight_tensor = _sdpo_rollout_is_weight_tensor(
        micro_batch,
        trainer_logprobs,
        sequence_lengths=sequence_lengths,
        sdpo_loss_config=sdpo_loss_config,
    )
    sdpo_rollout_is_weights = _validate_sdpo_rollout_is_weights(
        _optional_rollout_is_weight_tensor_to_floats(sdpo_rollout_is_weight_tensor, seq_len),
        sdpo_weights,
    )
    sdpo_student_topk_token_ids = _optional_tensor_rows_to_ints(
        model_output.get("sdpo_student_topk_token_ids"), seq_len, "sdpo_student_topk_token_ids"
    )
    sdpo_student_topk_logprobs = _optional_tensor_rows_to_logprobs(
        model_output.get("sdpo_student_topk_logprobs"), seq_len, "sdpo_student_topk_logprobs"
    )
    sdpo_topk_token_ids, sdpo_topk_logprobs = _mask_support_rows_to_sdpo_weights(
        sdpo_topk_token_ids, sdpo_topk_logprobs, sdpo_weights, "sdpo"
    )
    sdpo_student_topk_token_ids, sdpo_student_topk_logprobs = _mask_support_rows_to_sdpo_weights(
        sdpo_student_topk_token_ids, sdpo_student_topk_logprobs, sdpo_weights, "sdpo_student"
    )
    _validate_support_rows(sdpo_topk_token_ids, sdpo_topk_logprobs, "sdpo")
    _validate_support_rows(sdpo_student_topk_token_ids, sdpo_student_topk_logprobs, "sdpo_student")
    _validate_support_ownership(sdpo_topk_token_ids, sdpo_topk_logprobs, sdpo_weights, "sdpo")
    _validate_support_ownership(sdpo_student_topk_token_ids, sdpo_student_topk_logprobs, sdpo_weights, "sdpo_student")

    return {
        "token_ids": token_ids,
        "position_ids": _tensor_to_non_negative_ints(micro_batch["position_ids"], "position_ids"),
        "loss_mask": loss_mask,
        "temperatures": _tensor_to_floats(micro_batch["temperatures"], "temperatures"),
        "advantages": _tensor_to_floats(micro_batch["advantages"], "advantages"),
        "rewards": _optional_tensor_to_floats(micro_batch.get("rewards"), seq_len, "rewards"),
        "inference_logprobs": _tensor_to_logprob_floats(micro_batch["inference_logprobs"], "inference_logprobs"),
        "trainer_logprobs": _tensor_to_logprob_floats(trainer_logprobs, "trainer_logprobs"),
        "entropy": _tensor_to_logprob_floats(model_output["entropy"], "entropy"),
        "mismatch_kl": _optional_tensor_to_floats(export_tensors["mismatch_kl"], seq_len, "mismatch_kl"),
        "log_importance_ratio": _optional_tensor_to_floats(
            export_tensors["log_importance_ratio"], seq_len, "log_importance_ratio"
        ),
        "importance_ratio": _optional_tensor_to_floats(export_tensors["importance_ratio"], seq_len, "importance_ratio"),
        "prob_delta": _optional_tensor_to_floats(export_tensors["prob_delta"], seq_len, "prob_delta"),
        "is_masked": _optional_tensor_to_bools(export_tensors["is_masked"], seq_len, "is_masked"),
        "is_masked_high": _optional_tensor_to_bools(export_tensors["is_masked_high"], seq_len, "is_masked_high"),
        "is_masked_low": _optional_tensor_to_bools(export_tensors["is_masked_low"], seq_len, "is_masked_low"),
        # Component weight streams; ``None`` columns mean the defaults (rl 1.0
        # on the loss mask, no ce/ref_kl/sdpo component).
        "rl_weights": _optional_tensor_to_floats(micro_batch.get("rl_weights"), seq_len, "rl_weights"),
        "ce_weights": _optional_tensor_to_floats(micro_batch.get("ce_weights"), seq_len, "ce_weights"),
        "ref_kl_weights": _optional_tensor_to_floats(micro_batch.get("ref_kl_weights"), seq_len, "ref_kl_weights"),
        "sdpo_weights": sdpo_weights,
        "sdpo_rollout_is_weights": sdpo_rollout_is_weights,
        "sdpo_topk_token_ids": sdpo_topk_token_ids,
        "sdpo_topk_logprobs": sdpo_topk_logprobs,
        "sdpo_student_topk_token_ids": sdpo_student_topk_token_ids,
        "sdpo_student_topk_logprobs": sdpo_student_topk_logprobs,
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
    # (stream present but all-zero), no ref_kl member, and no sdpo member.
    rl_weights = micro_batch.get("rl_weights")
    ref_kl_weights = micro_batch.get("ref_kl_weights")
    sdpo_weights = micro_batch.get("sdpo_weights")
    no_rl = rl_weights is not None and not bool((rl_weights != 0).any())
    no_ref_kl = ref_kl_weights is None or not bool((ref_kl_weights != 0).any())
    no_sdpo = sdpo_weights is None or not bool(active_sdpo_weight_mask(sdpo_weights).any())
    if no_rl and no_ref_kl and no_sdpo:
        return fields

    _require_floating_tensor(trainer_logprobs, "trainer_logprobs")
    _require_floating_tensor(micro_batch["inference_logprobs"], "inference_logprobs")
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


def _sdpo_rollout_is_weight_tensor(
    micro_batch: Mapping[str, Any],
    trainer_logprobs: Tensor,
    *,
    sequence_lengths: Sequence[int],
    sdpo_loss_config: Any | None,
) -> Tensor | None:
    supplied = micro_batch.get("sdpo_rollout_is_weights")
    if supplied is not None:
        return supplied
    if sdpo_loss_config is None or getattr(sdpo_loss_config, "rollout_is", None) is None:
        return None
    if getattr(sdpo_loss_config, "rollout_is_batch_normalize", False):
        return None
    sdpo_weights = micro_batch.get("sdpo_weights")
    if sdpo_weights is None or not bool(active_sdpo_weight_mask(sdpo_weights).any()):
        return None
    _require_floating_tensor(trainer_logprobs, "trainer_logprobs")
    _require_floating_tensor(micro_batch["inference_logprobs"], "inference_logprobs")
    flat_trainer_logprobs = trainer_logprobs.reshape(-1)
    _check_sequence_lengths(sequence_lengths, int(flat_trainer_logprobs.numel()))
    flat_inference_logprobs = micro_batch["inference_logprobs"].to(trainer_logprobs.device).reshape(-1)
    flat_loss_mask = micro_batch["loss_mask"].to(trainer_logprobs.device).reshape(-1)
    flat_sdpo_mask = active_sdpo_weight_mask(sdpo_weights.to(trainer_logprobs.device)).reshape(-1)
    flat_rollout_is_weights = torch.zeros_like(flat_trainer_logprobs)
    start = 0
    for length in sequence_lengths:
        end = start + length
        response_mask = (flat_loss_mask[start:end] & flat_sdpo_mask[start:end]).unsqueeze(0)
        if bool(response_mask.any()):
            flat_rollout_is_weights[start:end] = compute_rollout_is_weights(
                log_ratio=(flat_trainer_logprobs[start:end] - flat_inference_logprobs[start:end]).detach().unsqueeze(0),
                response_mask=response_mask,
                rollout_is=sdpo_loss_config.rollout_is,
                rollout_is_threshold=sdpo_loss_config.rollout_is_threshold,
                rollout_is_batch_normalize=False,
            ).squeeze(0)
        start = end
    return flat_rollout_is_weights.reshape_as(trainer_logprobs)


def _preflight_only_flag(micro_batch: Mapping[str, Any]) -> bool:
    preflight_only = micro_batch.get("preflight_only", False)
    if not isinstance(preflight_only, bool):
        raise ValueError("preflight_only must be a boolean on token-export micro batches")
    return preflight_only


def _tensor_to_non_negative_ints(tensor: Tensor, name: str) -> list[int]:
    if torch.is_floating_point(tensor) or torch.is_complex(tensor) or tensor.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer token ids")
    if bool((tensor < 0).any()):
        raise ValueError(f"{name} must contain non-negative token ids")
    return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_to_bools(tensor: Tensor, name: str) -> list[bool]:
    if tensor.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor")
    return [bool(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_to_floats(tensor: Tensor, name: str) -> list[float | None]:
    if torch.is_complex(tensor) or tensor.dtype == torch.bool:
        raise ValueError(f"{name} must contain numeric values")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite values")
    values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
    return [_json_float(value) for value in values]


def _require_floating_tensor(tensor: Tensor, name: str) -> None:
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must use a floating-point dtype")


def _tensor_to_logprob_floats(tensor: Tensor, name: str) -> list[float | None]:
    _require_floating_tensor(tensor, name)
    return _tensor_to_floats(tensor, name)


def _optional_tensor_to_floats(tensor: Tensor | None, seq_len: int, name: str) -> list[float | None]:
    if tensor is None:
        return [None] * seq_len
    return _tensor_to_floats(tensor, name)


def _optional_weight_tensor_to_floats(tensor: Tensor | None, seq_len: int, name: str) -> list[float | None]:
    if tensor is None:
        return [None] * seq_len
    if tensor.numel() != seq_len:
        raise ValueError(f"{name} length {tensor.numel()} != sequence length {seq_len}")
    if torch.is_complex(tensor) or tensor.dtype == torch.bool:
        raise ValueError(f"{name} must contain numeric weights")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite weights")
    if bool((tensor < 0).any()):
        raise ValueError(f"{name} must contain non-negative weights")
    return _tensor_to_floats(tensor, name)


def _optional_rollout_is_weight_tensor_to_floats(tensor: Tensor | None, seq_len: int) -> list[float | None]:
    if tensor is not None and not torch.is_floating_point(tensor):
        raise ValueError("sdpo_rollout_is_weights must use a floating-point dtype")
    return _optional_weight_tensor_to_floats(tensor, seq_len, "sdpo_rollout_is_weights")


def _optional_tensor_to_bools(tensor: Tensor | None, seq_len: int, name: str) -> list[bool | None]:
    if tensor is None:
        return [None] * seq_len
    return _tensor_to_bools(tensor, name)


def _optional_tensor_rows_to_ints(tensor: Tensor | None, seq_len: int, name: str) -> list[list[int] | None]:
    if tensor is None:
        return [None] * seq_len
    rows = _as_token_rows(tensor, seq_len)
    if torch.is_floating_point(rows) or torch.is_complex(rows) or rows.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer token ids")
    if rows.shape[1] == 0:
        raise ValueError(f"{name} must contain non-empty top-k rows")
    if bool((rows < 0).any()):
        raise ValueError(f"{name} must contain non-negative token ids")
    return [[int(value) for value in row.tolist()] for row in rows]


def _optional_tensor_rows_to_floats(tensor: Tensor | None, seq_len: int, name: str) -> list[list[float] | None]:
    if tensor is None:
        return [None] * seq_len
    rows = _as_token_rows(tensor, seq_len)
    if torch.is_complex(rows) or rows.dtype == torch.bool:
        raise ValueError(f"{name} must contain numeric values")
    if rows.shape[1] == 0:
        raise ValueError(f"{name} must contain non-empty top-k rows")
    if not bool(torch.isfinite(rows).all()):
        raise ValueError(f"{name} must contain finite values")
    return [[float(value) for value in row.tolist()] for row in rows]


def _optional_tensor_rows_to_logprobs(tensor: Tensor | None, seq_len: int, name: str) -> list[list[float] | None]:
    if tensor is None:
        return [None] * seq_len
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must use a floating-point dtype")
    return _optional_tensor_rows_to_floats(tensor, seq_len, name)


def _mask_support_rows_to_sdpo_weights(
    token_rows: list[list[int] | None],
    logprob_rows: list[list[float] | None],
    weights: list[float | None],
    name: str,
) -> tuple[list[list[int] | None], list[list[float] | None]]:
    token_width = next((len(row) for row in token_rows if row is not None), None)
    logprob_width = next((len(row) for row in logprob_rows if row is not None), None)
    if token_width is None and logprob_width is None:
        return token_rows, logprob_rows
    if token_width is None or logprob_width is None:
        raise ValueError(f"{name} top-k token ids and logprobs must be exported as a pair")
    if token_width != logprob_width:
        raise ValueError(f"{name} top-k token ids/logprobs width mismatch: {token_width} != {logprob_width}")
    placeholder_ids = [0] * token_width
    placeholder_logprobs = [0.0] * token_width
    masked_token_rows: list[list[int] | None] = []
    masked_logprob_rows: list[list[float] | None] = []
    for weight, token_row, logprob_row in zip(weights, token_rows, logprob_rows, strict=True):
        if is_active_sdpo_weight(weight):
            masked_token_rows.append(token_row)
            masked_logprob_rows.append(logprob_row)
        else:
            masked_token_rows.append(list(placeholder_ids))
            masked_logprob_rows.append(list(placeholder_logprobs))
    return masked_token_rows, masked_logprob_rows


def _validate_sdpo_rollout_is_weights(
    rollout_is_weights: list[float | None],
    sdpo_weights: list[float | None],
) -> list[float | None]:
    for idx, (rollout_is_weight, sdpo_weight) in enumerate(zip(rollout_is_weights, sdpo_weights, strict=True)):
        if not is_active_sdpo_weight(rollout_is_weight):
            continue
        if not is_active_sdpo_weight(sdpo_weight):
            raise ValueError(f"sdpo_rollout_is_weights[{idx}] is nonzero outside SDPO component")
    return rollout_is_weights


def _validate_sdpo_weights_follow_loss_mask(sdpo_weights: Sequence[float | None], loss_mask: Sequence[bool]) -> None:
    for idx, (weight, trains) in enumerate(zip(sdpo_weights, loss_mask, strict=True)):
        if is_active_sdpo_weight(weight) and not trains:
            raise ValueError(f"sdpo_weights[{idx}] is nonzero outside loss_mask")


def _validate_support_rows(
    token_rows: Sequence[list[int] | None],
    logprob_rows: Sequence[list[float] | None],
    name: str,
) -> None:
    for idx, (token_row, logprob_row) in enumerate(zip(token_rows, logprob_rows, strict=True)):
        if token_row is None and logprob_row is None:
            continue
        if token_row is None or logprob_row is None:
            raise ValueError(f"{name} support row {idx} must include both token ids and logprobs")
        if _is_placeholder_logprob_row(logprob_row):
            if not _is_placeholder_token_id_row(token_row):
                raise ValueError(f"{name} top-k token ids row {idx} must be zero when logprobs are placeholders")
            continue
        if len(set(token_row)) != len(token_row):
            raise ValueError(f"{name} top-k token ids row {idx} must contain distinct token ids")
        row_mass = math.fsum(math.exp(logprob) for logprob in logprob_row)
        if row_mass > 1.0 + 1e-5:
            raise ValueError(f"{name} top-k logprobs row {idx} probability mass exceeds 1")


def _validate_support_ownership(
    token_rows: Sequence[list[int] | None],
    logprob_rows: Sequence[list[float] | None],
    weights: Sequence[float | None],
    name: str,
) -> None:
    if _has_nonzero_sdpo_weight(weights):
        return
    support_rows = zip(token_rows, logprob_rows, strict=True)
    if any(token_row is not None or logprob_row is not None for token_row, logprob_row in support_rows):
        raise ValueError(f"{name} top-k support requires nonzero sdpo_weights")


def _as_token_rows(tensor: Tensor, seq_len: int) -> Tensor:
    rows = tensor.detach().cpu()
    if rows.dim() == 3 and rows.shape[0] == 1:
        rows = rows.squeeze(0)
    if rows.dim() != 2 or rows.shape[0] != seq_len:
        raise ValueError(f"Expected per-token rows with shape [{seq_len}, width], got {list(rows.shape)}")
    return rows


def _check_lengths(columns: Mapping[str, Sequence[Any]]) -> None:
    lengths = {key: len(values) for key, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Token export fields must have aligned lengths, got {lengths}")


def _check_sequence_lengths(sequence_lengths: Sequence[int], total_length: int) -> None:
    if not sequence_lengths:
        raise ValueError("Token export sequence_lengths must contain at least one sequence")
    for idx, length in enumerate(sequence_lengths):
        if isinstance(length, bool) or not isinstance(length, int):
            raise ValueError(f"Token export sequence_lengths[{idx}] must be an integer")
        if length <= 0:
            raise ValueError(f"Token export sequence_lengths[{idx}] must be positive")
    sequence_total = sum(sequence_lengths)
    if sequence_total != total_length:
        raise ValueError(
            f"Token export sequence_lengths must sum to flattened token length "
            f"(sum={sequence_total}, tokens={total_length})"
        )


def _has_nonzero_sdpo_weight(weights: Sequence[float | None]) -> bool:
    return has_active_sdpo_weights(weights)


def _is_non_blank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_supported_sdpo_rows(
    weights: Sequence[float | None],
    token_rows: Sequence[list[int] | None],
    logprob_rows: Sequence[list[float] | None],
) -> bool:
    for weight, token_row, logprob_row in zip(weights, token_rows, logprob_rows, strict=True):
        if not is_active_sdpo_weight(weight):
            continue
        if token_row is None or logprob_row is None:
            continue
        if not _is_placeholder_logprob_row(logprob_row):
            return True
    return False


def _has_missing_or_placeholder_sdpo_rows(
    weights: Sequence[float | None],
    token_rows: Sequence[list[int] | None],
    logprob_rows: Sequence[list[float] | None],
) -> bool:
    for weight, token_row, logprob_row in zip(weights, token_rows, logprob_rows, strict=True):
        if not is_active_sdpo_weight(weight):
            continue
        if token_row is None or logprob_row is None:
            return True
        if _is_placeholder_logprob_row(logprob_row):
            return True
    return False


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


def _is_placeholder_logprob_row(logprobs: Sequence[float | None]) -> bool:
    return all(not isinstance(value, bool) and value in (0, 0.0) for value in logprobs)


def _is_placeholder_token_id_row(token_ids: Sequence[int | None]) -> bool:
    return all(not isinstance(value, bool) and value == 0 for value in token_ids)


def _first_non_empty(values: Sequence[str]) -> str | None:
    for value in values:
        if value:
            return value
    return None
