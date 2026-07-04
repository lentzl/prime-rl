import math
import os
import shutil
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence

from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.transport import (
    MicroBatch,
    MicroBatchSender,
    TrainingSample,
    TransportConfig,
    setup_micro_batch_sender,
    setup_training_batch_receiver,
)
from prime_rl.transport.sdpo import has_active_sdpo_weights, is_active_sdpo_weight
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_rollout_dir

TIMEOUT_SECONDS = 0.1
WATCHDOG_TIMEOUT_SECONDS = 1800  # 30 minutes


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_non_blank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_active_sdpo_component(sample: TrainingSample) -> bool:
    if sample.sdpo_weights is None:
        return False
    if not isinstance(sample.sdpo_weights, list):
        return False
    return has_active_sdpo_weights(sample.sdpo_weights)


def _is_placeholder_sdpo_logprob_row(logprob_row: Sequence[object]) -> bool:
    return all(not isinstance(logprob, bool) and float(logprob) == 0.0 for logprob in logprob_row)


def _is_placeholder_sdpo_token_row(token_row: Sequence[object]) -> bool:
    return all(not isinstance(token_id, bool) and token_id == 0 for token_id in token_row)


class BasePacker(ABC):
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfig,
        bin_cost: Callable[[Sequence[int]], int],
        start_step: int = 0,
    ):
        self.logger = get_logger()
        self.multi_run_manager = get_multi_run_manager()
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer = tokenizer
        self.bin_cost = bin_cost
        self.receiver = setup_training_batch_receiver(config)
        shutil.rmtree(get_rollout_dir(self.multi_run_manager.output_dir), ignore_errors=True)
        self.sender: MicroBatchSender = setup_micro_batch_sender(
            self.multi_run_manager.output_dir, dp_world_size, start_step, config
        )
        self._last_heartbeat = time.monotonic()
        self._watchdog_armed = threading.Event()
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

    def _heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def _arm_watchdog(self) -> None:
        self._last_heartbeat = time.monotonic()
        self._watchdog_armed.set()

    def _disarm_watchdog(self) -> None:
        self._watchdog_armed.clear()

    def _validate_sample(self, sample: TrainingSample, *, preflight_only: bool = False) -> tuple[bool, str | None]:
        """Validate a sample to ensure it won't crash or corrupt the trainer."""
        if not isinstance(sample.token_ids, list):
            return False, "Run wrote a sample with non-list token_ids"
        if not all(not isinstance(value, bool) and isinstance(value, int) for value in sample.token_ids):
            return False, "Run wrote a sample with non-integer token_ids"
        if not all(value >= 0 for value in sample.token_ids):
            return False, "Run wrote a sample with negative token_ids"
        sample_length = len(sample.token_ids)
        for name, arr in (
            ("mask", sample.mask),
            ("logprobs", sample.logprobs),
            ("temperatures", sample.temperatures),
            ("advantages", sample.advantages),
            ("rl_weights", sample.rl_weights),
            ("ce_weights", sample.ce_weights),
            ("ref_kl_weights", sample.ref_kl_weights),
            ("sdpo_weights", sample.sdpo_weights),
            ("sdpo_rollout_is_weights", sample.sdpo_rollout_is_weights),
        ):
            if arr is None:
                continue
            if not isinstance(arr, list):
                return False, f"Run wrote a sample with non-list {name}"
            if len(arr) != sample_length:
                return (
                    False,
                    f"Run wrote a sample with {name} length != token_ids length ({len(arr)} != {sample_length})",
                )
        valid, reason = self._validate_numeric_streams(sample)
        if not valid:
            return valid, reason
        valid, reason = self._validate_sample_identity(sample, preflight_only=preflight_only)
        if not valid:
            return valid, reason
        if sample_length == 0:
            return False, "Run wrote a sample with no tokens"
        if sample_length > self.seq_len:
            return (
                False,
                f"Run wrote a sample with length {sample_length} which exceeds max sequence length {self.seq_len}",
            )
        if sample.ref_logprobs is not None and len(sample.ref_logprobs) != sample_length:
            return (
                False,
                f"Run wrote a sample with ref logprobs length != sample length ({len(sample.ref_logprobs)} != {sample_length})",
            )
        position_ids = getattr(sample, "position_ids", None)
        if position_ids is not None:
            if not isinstance(position_ids, list):
                return False, "Run wrote a sample with non-list position_ids"
            if len(position_ids) != sample_length:
                return (
                    False,
                    f"Run wrote a sample with position_ids length != token_ids length "
                    f"({len(position_ids)} != {sample_length})",
                )
            if not all(not isinstance(value, bool) and isinstance(value, int) for value in position_ids):
                return False, "Run wrote a sample with non-integer position_ids"
            if not all(value >= 0 for value in position_ids):
                return False, "Run wrote a sample with negative position_ids"
        valid, reason = self._validate_sdpo_topk_streams(sample, sample_length)
        if not valid:
            return valid, reason
        valid, reason = self._validate_preflight_sdpo_target_support(sample, preflight_only=preflight_only)
        if not valid:
            return valid, reason
        valid, reason = self._validate_sdpo_topk_component_ownership(sample)
        if not valid:
            return valid, reason
        valid, reason = self._validate_sdpo_support_mask_alignment(sample)
        if not valid:
            return valid, reason
        valid, reason = self._validate_final_sdpo_target_support(sample, preflight_only=preflight_only)
        if not valid:
            return valid, reason
        return True, None

    def _validate_sample_identity(self, sample: TrainingSample, *, preflight_only: bool) -> tuple[bool, str | None]:
        if sample.sample_id is not None and not _is_non_blank_string(sample.sample_id):
            return False, "Run wrote a sample with malformed sample_id"
        if _has_active_sdpo_component(sample):
            if sample.sample_id is None:
                return False, "Run wrote an SDPO sample without sample_id"
            if not _is_non_blank_string(sample.env_name):
                return False, "Run wrote an SDPO sample without env_name"
        return True, None

    def _validate_batch_sample_identity(
        self, samples: Sequence[TrainingSample], *, preflight_only: bool
    ) -> tuple[bool, str | None]:
        seen_sample_ids: set[str] = set()
        phase = "preflight " if preflight_only else ""
        for sample in samples:
            if not _has_active_sdpo_component(sample):
                continue
            sample_id = sample.sample_id
            if not _is_non_blank_string(sample_id):
                continue
            if sample_id in seen_sample_ids:
                return (
                    False,
                    f"Run wrote multiple {phase}SDPO samples with duplicate sample_id {sample_id!r} "
                    "in the same run step",
                )
            seen_sample_ids.add(sample_id)
        return True, None

    def _validate_numeric_streams(self, sample: TrainingSample) -> tuple[bool, str | None]:
        if not all(isinstance(value, bool) for value in sample.mask):
            return False, "Run wrote a sample with non-boolean mask values"
        for name, arr in (
            ("logprobs", sample.logprobs),
            ("temperatures", sample.temperatures),
            ("advantages", sample.advantages),
            ("ref_logprobs", sample.ref_logprobs),
            ("rl_weights", sample.rl_weights),
            ("ce_weights", sample.ce_weights),
            ("ref_kl_weights", sample.ref_kl_weights),
            ("sdpo_weights", sample.sdpo_weights),
            ("sdpo_rollout_is_weights", sample.sdpo_rollout_is_weights),
        ):
            if arr is None:
                continue
            if not all(_is_finite_number(value) for value in arr):
                return False, f"Run wrote a sample with non-finite or non-numeric {name}"
        if not all(value > 0.0 for value in sample.temperatures):
            return False, "Run wrote a sample with non-positive temperatures"
        if sample.sdpo_weights is not None:
            for idx, (weight, trains) in enumerate(zip(sample.sdpo_weights, sample.mask, strict=True)):
                if weight < 0:
                    return False, f"Run wrote a sample with negative sdpo_weights at token {idx}"
                if is_active_sdpo_weight(weight) and not trains:
                    return False, f"Run wrote a sample with nonzero sdpo_weights outside mask at token {idx}"
        if sample.sdpo_rollout_is_weights is not None:
            for idx, rollout_is_weight in enumerate(sample.sdpo_rollout_is_weights):
                if rollout_is_weight < 0:
                    return False, f"Run wrote a sample with negative sdpo_rollout_is_weights at token {idx}"
                if not is_active_sdpo_weight(rollout_is_weight):
                    continue
                if not isinstance(rollout_is_weight, float):
                    return False, f"Run wrote a sample with non-floating sdpo_rollout_is_weights at token {idx}"
                if sample.sdpo_weights is None or not is_active_sdpo_weight(sample.sdpo_weights[idx]):
                    return (
                        False,
                        f"Run wrote a sample with nonzero sdpo_rollout_is_weights outside SDPO component at token {idx}",
                    )
        return True, None

    def _validate_sdpo_topk_streams(self, sample: TrainingSample, sample_length: int) -> tuple[bool, str | None]:
        token_ids = sample.sdpo_topk_token_ids
        logprobs = sample.sdpo_topk_logprobs
        if token_ids is None and logprobs is None:
            return True, None
        if token_ids is None or logprobs is None:
            missing = "sdpo_topk_token_ids" if token_ids is None else "sdpo_topk_logprobs"
            return False, f"Run wrote a sample with {missing} missing from the paired SDPO top-k streams"
        if not isinstance(token_ids, list):
            return False, "Run wrote a sample with non-list sdpo_topk_token_ids"
        if not isinstance(logprobs, list):
            return False, "Run wrote a sample with non-list sdpo_topk_logprobs"
        if len(token_ids) != sample_length:
            return (
                False,
                f"Run wrote a sample with sdpo_topk_token_ids length != token_ids length "
                f"({len(token_ids)} != {sample_length})",
            )
        if len(logprobs) != sample_length:
            return (
                False,
                f"Run wrote a sample with sdpo_topk_logprobs length != token_ids length "
                f"({len(logprobs)} != {sample_length})",
            )
        if sample_length == 0:
            return True, None

        if not isinstance(token_ids[0], list):
            return False, "Run wrote a sample with non-list sdpo_topk_token_ids row 0"
        width = len(token_ids[0])
        if width == 0:
            return False, "Run wrote a sample with empty SDPO top-k rows"
        for idx, (token_row, logprob_row) in enumerate(zip(token_ids, logprobs)):
            if not isinstance(token_row, list):
                return False, f"Run wrote a sample with non-list sdpo_topk_token_ids row {idx}"
            if not isinstance(logprob_row, list):
                return False, f"Run wrote a sample with non-list sdpo_topk_logprobs row {idx}"
            if len(token_row) != width:
                return (
                    False,
                    f"Run wrote a sample with ragged sdpo_topk_token_ids row {idx} ({len(token_row)} != {width})",
                )
            if len(logprob_row) != width:
                return (
                    False,
                    f"Run wrote a sample with sdpo_topk_logprobs row {idx} width != token row width "
                    f"({len(logprob_row)} != {width})",
                )
            if not all(not isinstance(value, bool) and isinstance(value, int) for value in token_row):
                return False, f"Run wrote a sample with non-integer sdpo_topk_token_ids row {idx}"
            if not all(value >= 0 for value in token_row):
                return False, f"Run wrote a sample with negative sdpo_topk_token_ids row {idx}"
            if not all(_is_finite_number(value) for value in logprob_row):
                return False, f"Run wrote a sample with non-finite or non-numeric sdpo_topk_logprobs row {idx}"
            if _is_placeholder_sdpo_logprob_row(logprob_row):
                if not _is_placeholder_sdpo_token_row(token_row):
                    return False, f"Run wrote a sample with nonzero placeholder sdpo_topk_token_ids row {idx}"
                continue
            if not all(isinstance(value, float) for value in logprob_row):
                return False, f"Run wrote a sample with non-floating sdpo_topk_logprobs row {idx}"
            if len(set(token_row)) != len(token_row):
                return False, f"Run wrote a sample with duplicate sdpo_topk_token_ids row {idx}"
        return True, None

    def _validate_preflight_sdpo_target_support(
        self, sample: TrainingSample, *, preflight_only: bool
    ) -> tuple[bool, str | None]:
        if not preflight_only:
            return True, None
        if sample.sdpo_topk_token_ids is None or sample.sdpo_topk_logprobs is None:
            return True, None
        for idx, weight in enumerate(sample.sdpo_weights or []):
            if not is_active_sdpo_weight(weight):
                continue
            token_row = sample.sdpo_topk_token_ids[idx]
            logprob_row = sample.sdpo_topk_logprobs[idx]
            if not _is_placeholder_sdpo_token_row(token_row) or not _is_placeholder_sdpo_logprob_row(logprob_row):
                return (
                    False,
                    f"Run wrote a preflight SDPO sample with transported top-k support at weighted token {idx}",
                )
        return True, None

    def _validate_sdpo_topk_component_ownership(self, sample: TrainingSample) -> tuple[bool, str | None]:
        if sample.sdpo_topk_token_ids is None and sample.sdpo_topk_logprobs is None:
            return True, None
        if _has_active_sdpo_component(sample):
            return True, None
        return False, "Run wrote SDPO top-k streams without nonzero sdpo_weights"

    def _validate_sdpo_support_mask_alignment(self, sample: TrainingSample) -> tuple[bool, str | None]:
        if sample.sdpo_weights is None or sample.sdpo_topk_token_ids is None or sample.sdpo_topk_logprobs is None:
            return True, None
        for idx, weight in enumerate(sample.sdpo_weights):
            if is_active_sdpo_weight(weight):
                continue
            token_row = sample.sdpo_topk_token_ids[idx]
            logprob_row = sample.sdpo_topk_logprobs[idx]
            if not _is_placeholder_sdpo_token_row(token_row) or not _is_placeholder_sdpo_logprob_row(logprob_row):
                return (
                    False,
                    f"Run wrote a sample with non-placeholder SDPO top-k support at unweighted token {idx}",
                )
        return True, None

    def _validate_final_sdpo_target_support(
        self, sample: TrainingSample, *, preflight_only: bool
    ) -> tuple[bool, str | None]:
        if preflight_only:
            return True, None
        if not has_active_sdpo_weights(sample.sdpo_weights or []):
            return True, None
        if sample.sdpo_topk_token_ids is None or sample.sdpo_topk_logprobs is None:
            return (
                False,
                "Run wrote a final SDPO sample with nonzero sdpo_weights but no transported SDPO top-k support",
            )
        for idx, weight in enumerate(sample.sdpo_weights or []):
            if not is_active_sdpo_weight(weight):
                continue
            if _is_placeholder_sdpo_logprob_row(sample.sdpo_topk_logprobs[idx]):
                return (
                    False,
                    f"Run wrote a final SDPO sample with placeholder top-k logprobs at weighted token {idx}",
                )
            row_mass = math.fsum(math.exp(float(logprob)) for logprob in sample.sdpo_topk_logprobs[idx])
            if row_mass > 1.0 + 1e-5:
                return (
                    False,
                    f"Run wrote a final SDPO sample with top-k logprob probability mass > 1 at weighted token {idx}",
                )
        return True, None

    def _watchdog_loop(self) -> None:
        while True:
            time.sleep(60)
            if not self._watchdog_armed.is_set():
                continue
            stale = time.monotonic() - self._last_heartbeat
            if stale > WATCHDOG_TIMEOUT_SECONDS:
                self.logger.error(f"Packer heartbeat stale for {stale:.0f}s, killing process to trigger restart")
                os._exit(1)

    @abstractmethod
    def pack(self) -> None:
        """Pack samples for the next step."""
        pass


class SinglePacker(BasePacker):
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfig,
        bin_cost: Callable[[Sequence[int]], int],
        start_step: int = 0,
    ):
        super().__init__(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, config, bin_cost, start_step)
        assert self.multi_run_manager.max_runs == 1, "SinglePacker only supports one run"

    def pack(self):
        # Wait for batch to be available
        batches = []
        while len(batches) == 0:
            self._heartbeat()
            self.multi_run_manager.discover_runs()
            batches = self.receiver.receive()
            time.sleep(0.2)

        assert len(batches) == 1, "SinglePacker only supports one batch per step"
        batch = batches[0]
        for sample in batch.examples:
            valid, reason = self._validate_sample(sample, preflight_only=batch.preflight_only)
            if not valid:
                raise ValueError(f"Run wrote a sample with invalid data: {reason}")
        valid, reason = self._validate_batch_sample_identity(batch.examples, preflight_only=batch.preflight_only)
        if not valid:
            raise ValueError(f"Run wrote a batch with invalid data: {reason}")

        if not batch.preflight_only:
            self.multi_run_manager.ready_to_update[0] = True
            self.multi_run_manager.progress[0].step += 1
        micro_batch_grid = prepare_batch(
            rollouts=batch.examples,
            seq_len=self.seq_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            num_train_workers=self.dp_world_size,
            idxs=[0] * len(batch.examples),
            num_loras=self.multi_run_manager.max_runs,
            bin_cost=self.bin_cost,
        )
        # The receiver always stamps run_idx from used_idxs (a key of idx_2_id).
        run_id = self.multi_run_manager.idx_2_id[batch.run_idx]
        for worker_batches in micro_batch_grid:
            for micro_batch in worker_batches:
                micro_batch.run_id = run_id
                micro_batch.run_step = batch.step
                micro_batch.preflight_only = batch.preflight_only
                micro_batch.preflight_step_complete = True

        self.sender.send(micro_batch_grid)


class MultiPacker(BasePacker):
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfig,
        bin_cost: Callable[[Sequence[int]], int],
        start_step: int = 0,
    ):
        super().__init__(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, config, bin_cost, start_step)
        # Per-run buffer: stores (TrainingSample, step, preflight_only) tuples.
        self.buffers: list[deque[tuple[TrainingSample, int, bool]]] = [
            deque() for _ in range(self.multi_run_manager.max_runs)
        ]

        # Round-robin position (persists across pack() calls)
        self._round_robin_position: int = 0

        # Register forgotten hook for receiver reset (master only, called during discover_runs)
        # This must happen when a run is deleted to prevent stale data from remaining
        self.multi_run_manager.register_forgotten_hook(self._on_run_data_deleted)

    def _on_run_data_deleted(self, idx: int, run_id: str) -> None:
        """Reset run state when run data is deleted (master only)."""
        self.logger.debug(f"Packing is resetting run state for deleted run {idx}")
        self.receiver.reset_run(idx)

        # Reset run state
        self.buffers[idx].clear()

    def _get_batch(self) -> None:
        """Receive batches from orchestrator and buffer samples per run."""
        self._heartbeat()
        self.multi_run_manager.discover_runs()
        batches = self.receiver.receive()

        for batch in batches:
            if batch.run_idx is None:
                self.logger.warning("Received batch with no run index")
                continue
            if len(batch.examples) == 0:
                self.multi_run_manager.evict_run(batch.run_idx, "Run wrote a batch with no samples")
                continue
            buffered_same_step = [
                sample
                for sample, step, preflight_only in self.buffers[batch.run_idx]
                if step == batch.step and preflight_only == batch.preflight_only
            ]
            valid, reason = self._validate_batch_sample_identity(
                [*buffered_same_step, *batch.examples],
                preflight_only=batch.preflight_only,
            )
            if not valid:
                self.multi_run_manager.evict_run(batch.run_idx, f"Run wrote a batch with invalid data: {reason}")
                continue
            for sample in batch.examples:
                valid, reason = self._validate_sample(sample, preflight_only=batch.preflight_only)
                if not valid:
                    self.multi_run_manager.evict_run(batch.run_idx, f"Run wrote a sample with invalid data: {reason}")
                    break
                self.buffers[batch.run_idx].append((sample, batch.step, batch.preflight_only))

        # This is necessary to forget evicted runs
        self.multi_run_manager.discover_runs()

    def _count_tokens(self, threshold: int | None = None, *, preflight_only: bool | None = None) -> int:
        tokens = 0

        for run_idx in self.multi_run_manager.used_idxs:
            buffer = self.buffers[run_idx]
            current_step = self.multi_run_manager.progress[run_idx].step

            for sample, step, sample_preflight_only in buffer:
                if step > current_step:
                    break
                if preflight_only is not None and sample_preflight_only != preflight_only:
                    continue
                tokens += len(sample.token_ids)
                if threshold is not None and tokens >= threshold:
                    return tokens
        return tokens

    def _has_enough_tokens(self) -> bool:
        """Check if we have enough samples in buffer to pack a step"""
        # When not using small batch granularity, require at least one full batch
        threshold = self.seq_len * self.dp_world_size
        # Preflight-only batches are a whole-step mode in the trainer and must
        # not be overtaken by final training batches already buffered from other
        # runs. If any preflight work is ready, use that mode for readiness.
        preflight_tokens = self._count_tokens(threshold, preflight_only=True)
        if preflight_tokens > 0:
            return preflight_tokens >= threshold
        return self._count_tokens(threshold, preflight_only=False) >= threshold

    def _select_samples_round_robin(self, token_budget: int) -> list[tuple[int, TrainingSample, int, bool]]:
        """Select samples using round-robin from runs with buffered work."""
        selected: list[tuple[int, TrainingSample, int, bool]] = []
        selected_preflight: bool | None = True if self._count_tokens(1, preflight_only=True) > 0 else None
        tokens_collected = 0

        while tokens_collected < token_budget:
            # Round-robin until we find a run with work for the current step
            for _ in range(len(self.buffers)):
                if len(self.buffers[self._round_robin_position]) > 0:
                    _, step, preflight_only = self.buffers[self._round_robin_position][0]
                    if step <= self.multi_run_manager.progress[self._round_robin_position].step and (
                        selected_preflight is None or preflight_only == selected_preflight
                    ):
                        break
                self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            else:
                # TODO: We could probably make the logic safer. This is basically counting on _has_enough_tokens() to be correct.
                # We also need to cover the timeout case here.
                break
            run_idx = self._round_robin_position
            self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            current_step = self.multi_run_manager.progress[run_idx].step

            while len(self.buffers[run_idx]) > 0:
                sample, step, preflight_only = self.buffers[run_idx][0]
                if step > current_step:
                    # Samples from different steps should be consumed later
                    break
                if selected_preflight is None:
                    selected_preflight = preflight_only
                elif preflight_only != selected_preflight:
                    break
                tokens_collected += len(sample.token_ids)
                if tokens_collected > token_budget:
                    if tokens_collected == (len(sample.token_ids)):
                        tokens_collected -= len(sample.token_ids)
                        # This means we have a sample that has more tokens than max seqlen
                        self.buffers[run_idx].popleft()
                        continue
                    return selected
                selected.append((run_idx, sample, step, preflight_only))
                self.buffers[run_idx].popleft()

        return selected

    def _update_run_progress(self, run_idx: int, num_samples: int, num_tokens: int, *, preflight_only: bool) -> None:
        """Update run progress; increment step when all samples from the current step have been consumed."""
        if preflight_only:
            return
        # HACK: This fixes the issue with branching rollouts having unpredictable batch size
        # However, it makes us unable to do incremental orchestrator rollouts
        # Removing the len(self.buffers[run_idx]) == 0 check would allow incremental orchestrator rollouts
        if (
            len(self.buffers[run_idx]) == 0
            or self.buffers[run_idx][0][1] > self.multi_run_manager.progress[run_idx].step
        ):
            self.multi_run_manager.progress[run_idx].step += 1
            self.multi_run_manager.ready_to_update[run_idx] = True

        self.multi_run_manager.progress[run_idx].total_tokens += num_tokens
        self.multi_run_manager.progress[run_idx].total_samples += num_samples

    def pack(self):
        """Pack samples from buffers using round-robin fair scheduling."""
        self._get_batch()
        start_time = time.time()

        while not self._has_enough_tokens():
            if time.time() - start_time > TIMEOUT_SECONDS and self._count_tokens() > 0:
                self.logger.warning("Timeout waiting for enough tokens to pack")
                break
            time.sleep(1)
            self._get_batch()

        token_budget = self.seq_len * self.dp_world_size
        selected_samples = self._select_samples_round_robin(token_budget)
        assert selected_samples, "No samples selected"

        # Group samples by run_idx - each microbatch must contain samples from only ONE run
        # because MultiLoRAGroupedExperts (MoE) only supports one adapter per microbatch
        samples_by_run: dict[int, list[TrainingSample]] = {}
        steps_by_run: dict[int, int] = {}
        preflight_by_run: dict[int, bool] = {}
        per_run_stats: dict[int, tuple[int, int]] = {}
        for run_idx, sample, step, preflight_only in selected_samples:
            if run_idx not in samples_by_run:
                samples_by_run[run_idx] = []
                steps_by_run[run_idx] = step
                preflight_by_run[run_idx] = preflight_only
            else:
                assert steps_by_run[run_idx] == step, "Micro batches for a run must come from a single run step"
                assert preflight_by_run[run_idx] == preflight_only, "Cannot mix preflight and train batches"
            samples_by_run[run_idx].append(sample)

            num_tokens = len(sample.token_ids)
            if run_idx in per_run_stats:
                cur_samples, cur_tokens = per_run_stats[run_idx]
                per_run_stats[run_idx] = (cur_samples + 1, cur_tokens + num_tokens)
            else:
                per_run_stats[run_idx] = (1, num_tokens)

        for run_idx, (num_samples, num_tokens) in per_run_stats.items():
            self._update_run_progress(
                run_idx,
                num_samples,
                num_tokens,
                preflight_only=preflight_by_run[run_idx],
            )
        preflight_complete_by_run = {
            run_idx: self._preflight_step_complete(
                run_idx,
                step=steps_by_run[run_idx],
                preflight_only=preflight_by_run[run_idx],
            )
            for run_idx in samples_by_run
        }

        # Pack each run separately to ensure no mixing of runs in microbatches
        all_micro_batches: list[list[MicroBatch]] = [[] for _ in range(self.dp_world_size)]
        for run_idx in sorted(samples_by_run.keys()):
            run_samples = samples_by_run[run_idx]
            run_micro_batch_grid = prepare_batch(
                rollouts=run_samples,
                seq_len=self.seq_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                num_train_workers=self.dp_world_size,
                idxs=[run_idx] * len(run_samples),
                num_loras=self.multi_run_manager.max_runs,
                bin_cost=self.bin_cost,
            )
            run_id = self.multi_run_manager.idx_2_id[run_idx]
            run_step = steps_by_run[run_idx]
            preflight_only = preflight_by_run[run_idx]
            # Merge into combined grid
            for worker_idx, worker_batches in enumerate(run_micro_batch_grid):
                for micro_batch in worker_batches:
                    micro_batch.run_id = run_id
                    micro_batch.run_step = run_step
                    micro_batch.preflight_only = preflight_only
                    micro_batch.preflight_step_complete = preflight_complete_by_run[run_idx]
                all_micro_batches[worker_idx].extend(worker_batches)

        self.sender.send(all_micro_batches)

    def _preflight_step_complete(self, run_idx: int, *, step: int, preflight_only: bool) -> bool:
        if not preflight_only:
            return True
        return not any(
            buffered_step == step and buffered_preflight
            for _sample, buffered_step, buffered_preflight in self.buffers[run_idx]
        )


def setup_packer(
    dp_world_size: int,
    seq_len: int,
    pad_to_multiple_of: int,
    tokenizer: PreTrainedTokenizer,
    transport_config: TransportConfig,
    bin_cost: Callable[[Sequence[int]], int],
    start_step: int = 0,
) -> BasePacker:
    multi_run_manager = get_multi_run_manager()
    if multi_run_manager.max_runs == 1:
        return SinglePacker(
            dp_world_size, seq_len, pad_to_multiple_of, tokenizer, transport_config, bin_cost, start_step
        )
    else:
        return MultiPacker(
            dp_world_size, seq_len, pad_to_multiple_of, tokenizer, transport_config, bin_cost, start_step
        )
