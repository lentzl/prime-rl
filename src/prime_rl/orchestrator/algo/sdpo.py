from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from itertools import cycle
from typing import TYPE_CHECKING
from uuid import UUID

from prime_rl.configs.algorithm import AlgorithmConfig, SDPOAlgorithmConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.sdpo_preflight import run_sdpo_student_support_preflight
from prime_rl.orchestrator.utils import (
    MAX_PREFILL_CANDIDATE_TOKEN_IDS,
    compute_prefill_candidate_logprobs,
    compute_prefill_topk_logprobs,
)
from prime_rl.transport.sdpo import has_active_sdpo_weights, is_active_sdpo_weight

if TYPE_CHECKING:
    from pathlib import Path

    from renderers.base import Renderer

    from prime_rl.orchestrator.types import RolloutView
    from prime_rl.transport import TrainingSample
    from prime_rl.transport.base import TrainingBatchSender
    from prime_rl.utils.client import InferencePool


SDPO_TEACHER_LIVE_ROLE = "sdpo_teacher"


def _is_placeholder_logprob_row(logprob_row: list[float]) -> bool:
    return all(not isinstance(logprob, bool) and float(logprob) == 0.0 for logprob in logprob_row)


class SDPOAlgorithm(Algorithm):
    """Self-distillation over feedback-conditioned teacher distributions.

    The policy samples rollouts; after filtering, each sampled assistant span
    is scored under a rewritten teacher context and ships a top-k distillation
    target to the trainer's ``sdpo`` component.
    """

    action_loss_type = "sdpo"
    model_role = "teacher"
    batch_preflight_name = "sdpo_student_support"

    def __init__(
        self,
        config: AlgorithmConfig,
        policy_pool: InferencePool,
        renderer: Renderer | None,
        live_pools: Mapping[str, InferencePool] | None = None,
    ):
        super().__init__(config, policy_pool, renderer, live_pools=live_pools)
        if not isinstance(config, SDPOAlgorithmConfig):
            raise TypeError("sdpo requires an SDPOAlgorithmConfig")
        if renderer is None:
            raise ValueError("sdpo requires the renderer")
        self.teacher = config.model
        self.teacher_regularization = config.teacher_regularization
        self.teacher_update_rate = config.teacher_update_rate
        self.distillation_topk = config.distillation_topk
        self.distillation_topk_support = config.distillation_topk_support
        self.preflight_export_timeout_s = config.preflight_export_timeout_s
        self.success_reward_threshold = config.success_reward_threshold
        self.successful_demonstration_selection = config.successful_demonstration_selection
        self.dont_reprompt_on_self_success = config.dont_reprompt_on_self_success
        self.remove_thinking_from_demonstration = config.remove_thinking_from_demonstration
        self.include_environment_feedback = config.include_environment_feedback
        self.environment_feedback_only_without_solution = config.environment_feedback_only_without_solution
        self.max_reprompt_len = config.max_reprompt_len
        self.reprompt_truncation = config.reprompt_truncation
        self.template = config.template
        self.template_target = config.template_target
        self.solution_template = config.solution_template
        self.feedback_template = config.feedback_template
        self.assistant_prefix = config.assistant_prefix
        self.max_concurrent = config.max_concurrent
        self.multi_turn = config.multi_turn
        self.teacher_pool: InferencePool | None = None

    async def setup(self) -> None:
        if self.teacher_regularization == "live-policy":
            self.teacher_pool = await self.connect(self.teacher)
        else:
            self.teacher_pool = await self.connect_live(SDPO_TEACHER_LIVE_ROLE)

    def needs_batch_preflight(self, batch: list[RolloutView]) -> bool:
        if self.distillation_topk_support != "student":
            return False
        for rollout in batch:
            for sample in rollout.samples:
                if sample.sdpo_weights is None:
                    if any(sample.mask):
                        return True
                    continue
                if has_active_sdpo_weights(sample.sdpo_weights):
                    return True
        return False

    def batch_preflight_signature(self) -> object:
        return (
            "sdpo_student_support",
            "support=student",
            "sample_ids=strict",
            self.distillation_topk,
            self.preflight_export_timeout_s,
        )

    async def run_batch_preflight(
        self,
        batch: list[RolloutView],
        *,
        samples: list[TrainingSample],
        output_dir: Path,
        sender: TrainingBatchSender,
        step: int,
    ) -> None:
        if not samples:
            return
        await run_sdpo_student_support_preflight(
            output_dir=output_dir,
            sender=sender,
            samples=samples,
            step=step,
            expected_topk=self.distillation_topk,
            export_timeout_s=self.preflight_export_timeout_s,
        )

    def _is_successful(self, rollout: RolloutView) -> bool:
        return rollout.reward >= self.success_reward_threshold

    def _sampled_text(self, rollout: RolloutView) -> str:
        pieces: list[str] = []
        for branch in rollout.raw.branches:
            if not any(branch.sampled_mask):
                continue
            for node in branch.nodes:
                if node.sampled:
                    content = node.message.content
                    if isinstance(content, str):
                        pieces.append(content)
            if pieces:
                break
        return "\n".join(pieces)

    def _remove_thinking_trace(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    def _successful_previous_rollouts(self, batch: list[RolloutView]) -> dict[UUID, list[tuple[RolloutView, str]]]:
        successful: dict[UUID, list[tuple[RolloutView, str]]] = {}
        for rollout in batch:
            if not self._is_successful(rollout):
                continue
            group_id = rollout.raw.group_id
            raw_text = self._sampled_text(rollout)
            text = raw_text
            if self.remove_thinking_from_demonstration:
                text = self._remove_thinking_trace(text)
            successful.setdefault(group_id, []).append((rollout, text))
        if self.successful_demonstration_selection == "highest_reward":
            for candidates in successful.values():
                candidates.sort(key=lambda candidate: candidate[0].reward, reverse=True)
        return successful

    def _solution_for_rollout(
        self, rollout: RolloutView, successful_by_group: dict[UUID, list[tuple[RolloutView, str]]]
    ) -> str | None:
        candidates = successful_by_group.get(rollout.raw.group_id, [])
        if self.dont_reprompt_on_self_success:
            candidates = [(candidate, text) for candidate, text in candidates if candidate.raw is not rollout.raw]
        if not candidates:
            return None
        return candidates[0][1]

    def _skip_self_success_without_solution(self, rollout: RolloutView, solution: str | None) -> bool:
        return self.dont_reprompt_on_self_success and solution is None and self._is_successful(rollout)

    def _use_feedback(self, *, solution: str | None, feedback: str) -> bool:
        if not self.include_environment_feedback or not feedback:
            return False
        return not (self.environment_feedback_only_without_solution and solution is not None)

    def _has_sdpo_weight(self, sample: TrainingSample) -> bool:
        return has_active_sdpo_weights(sample.sdpo_weights or [])

    def _single_turn_trainable_branch(self, rollout: RolloutView):
        trainable_branches = [branch for branch in rollout.raw.branches if any(branch.sampled_mask)]
        if len(trainable_branches) != 1:
            raise ValueError(
                f"sdpo expected exactly one trainable branch for single-turn rollout "
                f"(env '{rollout.env_name}', trainable_branches={len(trainable_branches)})."
            )
        return trainable_branches[0]

    def select_batch_preflight_samples(
        self,
        batch: list[RolloutView],
        *,
        samples: list[TrainingSample],
    ) -> list[TrainingSample]:
        allowed_sample_ids = {id(sample) for sample in samples}
        return self._student_support_preflight_samples(batch, allowed_sample_ids=allowed_sample_ids)

    def _ensure_sdpo_weights(self, sample: TrainingSample) -> None:
        if sample.sdpo_weights is None:
            if not isinstance(sample.mask, list):
                raise ValueError(
                    f"sample mask must be a list before defaulting sdpo_weights "
                    f"(env '{sample.env_name}', got={type(sample.mask).__name__})."
                )
            sample.sdpo_weights = [1.0 if trains else 0.0 for trains in sample.mask]
        self._validate_sdpo_weights(sample)

    def _validate_sdpo_weights(self, sample: TrainingSample) -> None:
        if sample.sdpo_weights is None:
            raise ValueError(f"sdpo_weights must be initialized before validation (env '{sample.env_name}').")
        if not isinstance(sample.token_ids, list):
            raise ValueError(
                f"sample token_ids must be a list before validating sdpo_weights "
                f"(env '{sample.env_name}', got={type(sample.token_ids).__name__})."
            )
        if not isinstance(sample.mask, list):
            raise ValueError(
                f"sample mask must be a list before validating sdpo_weights "
                f"(env '{sample.env_name}', got={type(sample.mask).__name__})."
            )
        if not isinstance(sample.sdpo_weights, list):
            raise ValueError(
                f"sdpo_weights must be a list before validation "
                f"(env '{sample.env_name}', got={type(sample.sdpo_weights).__name__})."
            )
        if len(sample.mask) != len(sample.token_ids):
            raise ValueError(
                f"sample mask length must match token_ids length "
                f"(env '{sample.env_name}', mask={len(sample.mask)}, tokens={len(sample.token_ids)})."
            )
        if len(sample.sdpo_weights) != len(sample.token_ids):
            raise ValueError(
                f"sdpo_weights length must match token_ids length "
                f"(env '{sample.env_name}', weights={len(sample.sdpo_weights)}, tokens={len(sample.token_ids)})."
            )
        for idx, (weight, trains) in enumerate(zip(sample.sdpo_weights, sample.mask, strict=True)):
            if not isinstance(trains, bool):
                raise ValueError(f"sample mask must contain booleans (env '{sample.env_name}', token={idx}).")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
                raise ValueError(
                    f"sdpo_weights must contain finite numeric values (env '{sample.env_name}', token={idx})."
                )
            if weight < 0:
                raise ValueError(f"sdpo_weights must be non-negative (env '{sample.env_name}', token={idx}).")
            if is_active_sdpo_weight(weight) and not trains:
                raise ValueError(
                    f"sdpo_weights must be zero outside sampled tokens (env '{sample.env_name}', token={idx})."
                )

    def _sdpo_target_positions(self, sample: TrainingSample, positions: list[int] | None = None) -> list[int]:
        """Positions included in the self-distillation mask.

        ``sample.mask`` says which tokens are model-sampled; ``sdpo_weights``
        is the SDPO component/self-distillation mask. Teacher scoring should
        follow the latter so preflight/export evidence describes actual SDPO
        targets, not merely trainable-but-pruned tokens.
        """
        self._ensure_sdpo_weights(sample)
        if sample.sdpo_weights is None:
            raise ValueError(
                f"sdpo_weights must be initialized before selecting target positions (env '{sample.env_name}')."
            )
        candidates: range | list[int]
        if positions is None:
            candidates = range(len(sample.mask))
        else:
            seen: set[int] = set()
            for position in positions:
                if isinstance(position, bool) or not isinstance(position, int):
                    raise ValueError(
                        f"sdpo target positions must be integer token indices "
                        f"(env '{sample.env_name}', got={type(position).__name__})."
                    )
                if position < 0 or position >= len(sample.mask):
                    raise ValueError(
                        f"sdpo target position outside token range "
                        f"(env '{sample.env_name}', position={position}, tokens={len(sample.mask)})."
                    )
                if position in seen:
                    raise ValueError(
                        f"sdpo target positions must be unique (env '{sample.env_name}', position={position})."
                    )
                seen.add(position)
            candidates = positions
        return [i for i in candidates if sample.mask[i] and is_active_sdpo_weight(sample.sdpo_weights[i])]

    def _student_support_preflight_samples(
        self, batch: list[RolloutView], *, allowed_sample_ids: set[int]
    ) -> list[TrainingSample]:
        """Return samples whose final SDPO target needs student-selected support.

        This mirrors the hindsight-target gating in ``score_batch`` so
        preflight-only token exports prove actual SDPO targets, not merely
        samples that were routed to the SDPO component before hindsight
        filtering.
        """
        successful_by_group = self._successful_previous_rollouts(batch)
        preflight_samples: list[TrainingSample] = []

        def sample_allowed(sample: TrainingSample) -> bool:
            return id(sample) in allowed_sample_ids

        for rollout in batch:
            solution = self._solution_for_rollout(rollout, successful_by_group)
            if self._skip_self_success_without_solution(rollout, solution):
                for sample in rollout.samples:
                    if sample_allowed(sample):
                        self._clear_sdpo_target(sample)
                continue
            if rollout.raw.num_turns == 1:
                if len(rollout.samples) != 1:
                    raise ValueError(
                        f"sdpo expected exactly one sample for single-turn rollout "
                        f"(env '{rollout.env_name}', samples={len(rollout.samples)})."
                    )
                branch = self._single_turn_trainable_branch(rollout)
                sample = rollout.samples[0]
                if not sample_allowed(sample):
                    continue
                self._validate_branch_sample_alignment(branch, sample, rollout.env_name)
                sampled_node_index = self._single_sampled_node_index(branch, rollout.env_name)
                raw_feedback = self._single_turn_feedback(rollout, branch, sampled_node_index)
                feedback = raw_feedback if self._use_feedback(solution=solution, feedback=raw_feedback) else ""
                if solution is None and not feedback:
                    self._clear_sdpo_target(sample)
                    continue
                self._ensure_sdpo_weights(sample)
                if self._has_sdpo_weight(sample):
                    preflight_samples.append(sample)
                continue
            if not self.multi_turn:
                raise ValueError(
                    f"sdpo supports single-step trajectories by default; "
                    f"env '{rollout.env_name}' produced {rollout.raw.num_turns} model turn(s)."
                )
            trainable_branches = [branch for branch in rollout.raw.branches if any(branch.sampled_mask)]
            if len(trainable_branches) != len(rollout.samples):
                raise ValueError(
                    f"sdpo expected one sample per trainable branch; got {len(rollout.samples)} sample(s) "
                    f"for {len(trainable_branches)} branch(es) (env '{rollout.env_name}')."
                )
            for branch, sample in zip(trainable_branches, rollout.samples):
                if not sample_allowed(sample):
                    continue
                self._validate_branch_sample_alignment(branch, sample, rollout.env_name)
                self._ensure_sdpo_weights(sample)
                offset = 0
                for node_index, node in enumerate(branch.nodes):
                    span = range(offset, offset + len(node.token_ids))
                    positions = self._sdpo_target_positions(sample, [i for i in span if sample.mask[i]])
                    if node.sampled and positions:
                        raw_feedback = self._feedback_after_node(branch, node_index)
                        feedback = raw_feedback if self._use_feedback(solution=solution, feedback=raw_feedback) else ""
                        if solution is None and not feedback:
                            self._zero_sdpo_positions(sample, positions)
                    offset += len(node.token_ids)
                if self._has_sdpo_weight(sample):
                    preflight_samples.append(sample)
                else:
                    self._clear_sdpo_target(sample)
        return preflight_samples

    def _feedback_after_node(self, branch, node_index: int) -> str:
        feedback: list[str] = []
        for node in branch.nodes[node_index + 1 :]:
            if node.sampled:
                break
            content = node.message.content
            if isinstance(content, str) and content.strip():
                feedback.append(f"{node.message.role}: {content}")
        return "\n".join(feedback)

    def _single_turn_feedback(self, rollout: RolloutView, branch, sampled_node_index: int) -> str:
        feedback = rollout.raw.info.get("feedback")
        if isinstance(feedback, str) and feedback.strip():
            return feedback
        return self._feedback_after_node(branch, sampled_node_index)

    def _teacher_prefix_ids(
        self,
        messages: list[dict],
        *,
        successful_previous_rollout: str | None,
        hindsight_feedback: str,
        env_name: str,
    ) -> list[int]:
        messages = [dict(message) for message in messages]
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            raise ValueError(f"sdpo found no user message to condition (env '{env_name}').")
        user_index = user_indices[0] if self.template_target == "first_user" else user_indices[-1]
        user_message = messages[user_index]
        question = user_message.get("content")
        if not isinstance(question, str):
            raise ValueError("sdpo supports text-only prompts (user content must be a string).")

        successful_solution_block = ""
        feedback_block = ""
        if successful_previous_rollout is not None:
            successful_solution_block = self.solution_template.format(
                successful_previous_attempt=successful_previous_rollout
            )
        if hindsight_feedback:
            feedback_block = self.feedback_template.format(feedback_raw=hindsight_feedback)
        user_message["content"] = self.template.format(
            question=question,
            successful_previous_rollout=successful_previous_rollout or "",
            hindsight_feedback=hindsight_feedback,
            successful_solution_block=successful_solution_block,
            feedback_block=feedback_block,
        )

        if self.assistant_prefix:
            prefix_ids = self._assistant_prefill_ids(messages, self.assistant_prefix)
        else:
            renderer = self._require_renderer()
            prefix_ids = list(renderer.render_ids(messages, add_generation_prompt=True))
        prefix_ids = self._truncate_reprompt(prefix_ids, env_name)
        if not prefix_ids:
            raise ValueError(f"sdpo teacher reprompt rendered to an empty token prefix (env '{env_name}').")
        return prefix_ids

    def _truncate_reprompt(self, prefix_ids: list[int], env_name: str) -> list[int]:
        if len(prefix_ids) <= self.max_reprompt_len:
            return prefix_ids
        if self.reprompt_truncation == "left":
            return prefix_ids[-self.max_reprompt_len :]
        if self.reprompt_truncation == "right":
            return prefix_ids[: self.max_reprompt_len]
        raise ValueError(
            f"sdpo teacher reprompt exceeds max_reprompt_len={self.max_reprompt_len} "
            f"with reprompt_truncation='error' (env '{env_name}', length={len(prefix_ids)})."
        )

    def _assistant_prefill_ids(self, messages: list[dict], text: str) -> list[int]:
        renderer = self._require_renderer()
        render_assistant_prefill_ids = getattr(renderer, "render_assistant_prefill_ids", None)
        if render_assistant_prefill_ids is not None:
            return list(render_assistant_prefill_ids(messages, text))
        rendered_prefill_ids = self._render_assistant_message_prefill_ids(messages, text)
        if rendered_prefill_ids is not None:
            return rendered_prefill_ids
        return renderer.render_ids(messages, add_generation_prompt=True) + self._text_ids(text)

    def _render_assistant_message_prefill_ids(self, messages: list[dict], text: str) -> list[int] | None:
        renderer = self._require_renderer()
        render = getattr(renderer, "render", None)
        get_stop_token_ids = getattr(renderer, "get_stop_token_ids", None)
        if render is None or get_stop_token_ids is None:
            return None

        assistant_idx = len(messages)
        rendered = render([*messages, {"role": "assistant", "content": text}], add_generation_prompt=False)
        token_ids = list(rendered.token_ids)
        sampled_mask = list(getattr(rendered, "sampled_mask", []))
        message_indices = list(getattr(rendered, "message_indices", []))
        if not sampled_mask:
            return None
        if len(sampled_mask) != len(token_ids) or len(message_indices) != len(token_ids):
            raise ValueError("sdpo renderer attribution must align with rendered token ids.")

        end = len(token_ids)
        while end > 0 and not sampled_mask[end - 1]:
            end -= 1
        stop_ids = set(get_stop_token_ids())
        if end == 0 or message_indices[end - 1] != assistant_idx or token_ids[end - 1] not in stop_ids:
            raise ValueError("sdpo renderer could not identify the assistant-prefill close token.")
        return token_ids[: end - 1]

    def _text_ids(self, text: str) -> list[int]:
        if not text:
            return []
        renderer = self._require_renderer()
        encode_text = getattr(renderer, "encode_text", None)
        if encode_text is not None:
            return list(encode_text(text))
        tokenizer = getattr(renderer, "tokenizer", None)
        if tokenizer is not None:
            return list(tokenizer.encode(text, add_special_tokens=False))
        raise ValueError("sdpo assistant_prefix requires renderer.encode_text or renderer.tokenizer.encode")

    def _require_renderer(self) -> Renderer:
        if self.renderer is None:
            raise RuntimeError("sdpo requires a renderer; construct SDPOAlgorithm with a renderer before scoring.")
        return self.renderer

    def _validate_branch_sample_alignment(self, branch, sample: TrainingSample, env_name: str) -> None:
        if list(sample.token_ids) != list(branch.token_ids):
            raise ValueError(
                f"sdpo expected sample tokens to align with the trace branch "
                f"(env '{env_name}', sample={len(sample.token_ids)}, branch={len(branch.token_ids)})."
            )
        if len(sample.mask) != len(branch.sampled_mask):
            raise ValueError(
                f"sdpo expected sample mask to align with the trace branch "
                f"(env '{env_name}', sample={len(sample.mask)}, branch={len(branch.sampled_mask)})."
            )
        if list(sample.mask) != list(branch.sampled_mask):
            raise ValueError(f"sdpo expected sample mask values to match the trace branch (env '{env_name}').")

    def _single_sampled_node_index(self, branch, env_name: str) -> int:
        sampled_indices = [i for i, node in enumerate(branch.nodes) if node.sampled]
        if len(sampled_indices) != 1:
            raise ValueError(
                f"sdpo expected exactly one sampled node for single-turn rollout "
                f"(env '{env_name}', sampled_nodes={len(sampled_indices)})."
            )
        return sampled_indices[0]

    def _clear_sdpo_target(self, sample: TrainingSample) -> None:
        sample.sdpo_weights = [0.0] * len(sample.token_ids)
        sample.sdpo_topk_token_ids = None
        sample.sdpo_topk_logprobs = None
        sample.sdpo_rollout_is_weights = None

    def _zero_sdpo_positions(self, sample: TrainingSample, positions: list[int]) -> None:
        self._ensure_sdpo_weights(sample)
        if sample.sdpo_weights is None:
            raise ValueError(
                f"sdpo_weights must be initialized before zeroing target positions (env '{sample.env_name}')."
            )
        for i in positions:
            sample.sdpo_weights[i] = 0.0
            if sample.sdpo_rollout_is_weights is not None:
                sample.sdpo_rollout_is_weights[i] = 0.0
            if sample.sdpo_topk_token_ids is not None:
                sample.sdpo_topk_token_ids[i] = [0] * self.distillation_topk
            if sample.sdpo_topk_logprobs is not None:
                sample.sdpo_topk_logprobs[i] = [0.0] * self.distillation_topk

    def _reject_precomputed_rollout_is_weights(self, sample: TrainingSample, positions: list[int]) -> None:
        if not positions or sample.sdpo_rollout_is_weights is None:
            return
        raise ValueError(
            "sdpo scoring expects rollout-IS weights to be derived by the trainer; "
            f"sample already carries sdpo_rollout_is_weights (env '{sample.env_name}')."
        )

    async def score_batch(self, batch: list[RolloutView]) -> None:
        pool = self.teacher_pool
        if pool is None:
            raise RuntimeError("sdpo teacher pool is not connected; Algorithm.setup() must run before score_batch().")
        if not batch:
            return
        train_clients = list(pool.train_clients)
        if not train_clients:
            raise RuntimeError("sdpo teacher pool has no train clients configured for feedback-conditioned scoring.")
        semaphore = asyncio.Semaphore(self.max_concurrent)
        successful_by_group = self._successful_previous_rollouts(batch)

        async def score_teacher_topk_completion(
            client, prefix_ids: list[int], completion_ids: list[int], target_offsets: list[int]
        ) -> tuple[list[list[int]], list[list[float]]]:
            async with semaphore:
                topk_ids, topk_logprobs = await compute_prefill_topk_logprobs(
                    client,
                    pool.model_name,
                    prefix_ids + completion_ids,
                    self.distillation_topk,
                )
            completion_topk_ids = topk_ids[-len(completion_ids) :]
            completion_topk_logprobs = topk_logprobs[-len(completion_ids) :]
            if len(completion_topk_ids) != len(completion_ids) or len(completion_topk_logprobs) != len(completion_ids):
                raise ValueError(
                    f"sdpo teacher top-k row count mismatch (expected={len(completion_ids)}, "
                    f"token_rows={len(completion_topk_ids)}, logprob_rows={len(completion_topk_logprobs)})."
                )
            for offset in target_offsets:
                if offset < 0 or offset >= len(completion_ids):
                    raise ValueError(
                        f"sdpo teacher target offset {offset} outside completion length {len(completion_ids)}."
                    )
            return (
                [completion_topk_ids[offset] for offset in target_offsets],
                [completion_topk_logprobs[offset] for offset in target_offsets],
            )

        async def score_student_support_completion(
            client,
            prefix_ids: list[int],
            completion_ids: list[int],
            target_offsets: list[int],
            candidate_rows: list[list[int]],
        ) -> list[list[float]]:
            if len(candidate_rows) != len(target_offsets):
                raise ValueError(
                    f"sdpo student-support row count mismatch (expected={len(target_offsets)}, "
                    f"got={len(candidate_rows)})."
                )
            for offset in target_offsets:
                if offset < 0 or offset >= len(completion_ids):
                    raise ValueError(
                        f"sdpo student-support target offset {offset} outside completion length {len(completion_ids)}."
                    )
            scored_rows: list[list[float]] = []
            for start, end in self._student_support_chunks(candidate_rows):
                # Keep the left context exact: for chunk [start:end), the
                # teacher still prefills prefix plus all earlier completion
                # tokens, but only returns candidate logprobs for this chunk.
                context_len = target_offsets[end - 1] + 1
                token_context = prefix_ids + completion_ids[:context_len]
                candidates = [[] for _ in token_context]
                for row_idx in range(start, end):
                    candidates[len(prefix_ids) + target_offsets[row_idx]] = candidate_rows[row_idx]
                async with semaphore:
                    logprob_rows = await compute_prefill_candidate_logprobs(
                        client,
                        pool.model_name,
                        token_context,
                        candidates,
                    )
                if len(logprob_rows) != len(token_context):
                    raise ValueError(
                        f"sdpo student-support teacher response row count mismatch "
                        f"(expected={len(token_context)}, got={len(logprob_rows)})."
                    )
                for row_idx in range(start, end):
                    response_row = logprob_rows[len(prefix_ids) + target_offsets[row_idx]]
                    if not isinstance(response_row, list):
                        raise ValueError(
                            f"sdpo student-support teacher response row must be a list "
                            f"(row={row_idx}, got={type(response_row).__name__})."
                        )
                    if len(response_row) != len(candidate_rows[row_idx]):
                        raise ValueError(
                            f"sdpo student-support teacher response row width mismatch "
                            f"(row={row_idx}, expected={len(candidate_rows[row_idx])}, got={len(response_row)})."
                        )
                scored_rows.extend(
                    logprob_rows[len(prefix_ids) + target_offsets[row_idx]] for row_idx in range(start, end)
                )
            return scored_rows

        def candidate_rows_from_sample(
            sample: TrainingSample,
            positions: list[int],
            env_name: str,
            *,
            token_rows: list[list[int]] | None = None,
        ) -> list[list[int]]:
            support_rows = sample.sdpo_topk_token_ids if token_rows is None else token_rows
            if support_rows is None:
                raise ValueError(
                    "sdpo distillation_topk_support='student' requires hydrated student-selected "
                    f"support from the preflight token export before teacher rescoring (env '{env_name}')."
                )
            if len(support_rows) != len(sample.token_ids):
                raise ValueError(
                    f"sdpo student top-k stream length must match token_ids length "
                    f"(env '{env_name}', rows={len(support_rows)}, tokens={len(sample.token_ids)})."
                )
            rows = [support_rows[i] for i in positions]
            self._validate_token_rows(rows, len(positions), env_name)
            return rows

        async def score_single_turn(client, rollout: RolloutView) -> None:
            if len(rollout.samples) != 1:
                raise ValueError(
                    f"sdpo expected exactly one sample for single-turn rollout "
                    f"(env '{rollout.env_name}', samples={len(rollout.samples)})."
                )
            branch = self._single_turn_trainable_branch(rollout)
            sample = rollout.samples[0]
            self._validate_branch_sample_alignment(branch, sample, rollout.env_name)
            solution = self._solution_for_rollout(rollout, successful_by_group)
            if self._skip_self_success_without_solution(rollout, solution):
                self._clear_sdpo_target(sample)
                return
            sampled_node_index = self._single_sampled_node_index(branch, rollout.env_name)
            sampled_node_start = sum(len(node.token_ids) for node in branch.nodes[:sampled_node_index])
            sampled_node = branch.nodes[sampled_node_index]
            sampled_span = list(range(sampled_node_start, sampled_node_start + len(sampled_node.token_ids)))
            raw_feedback = self._single_turn_feedback(rollout, branch, sampled_node_index)
            feedback = raw_feedback if self._use_feedback(solution=solution, feedback=raw_feedback) else ""
            if solution is None and not feedback:
                self._clear_sdpo_target(sample)
                return
            messages = [
                node.message.model_dump(exclude_none=True)
                for node in branch.nodes[:sampled_node_index]
                if not node.sampled
            ]
            prefix_ids = self._teacher_prefix_ids(
                messages,
                successful_previous_rollout=solution,
                hindsight_feedback=feedback,
                env_name=rollout.env_name,
            )
            positions = self._sdpo_target_positions(sample, sampled_span)
            if not positions:
                self._clear_sdpo_target(sample)
                return
            self._reject_precomputed_rollout_is_weights(sample, positions)
            target_offsets = [i - sampled_node_start for i in positions]
            completion_ids = list(sampled_node.token_ids)
            if self.distillation_topk_support == "student":
                token_rows = candidate_rows_from_sample(sample, positions, rollout.env_name)
                logprob_rows = await score_student_support_completion(
                    client, prefix_ids, completion_ids, target_offsets, token_rows
                )
            else:
                token_rows, logprob_rows = await score_teacher_topk_completion(
                    client, prefix_ids, completion_ids, target_offsets
                )
            self._scatter_topk(sample, token_rows, logprob_rows, positions=positions)

        async def score_multi_turn(client, rollout: RolloutView) -> None:
            solution = self._solution_for_rollout(rollout, successful_by_group)
            if self._skip_self_success_without_solution(rollout, solution):
                for sample in rollout.samples:
                    self._clear_sdpo_target(sample)
                return
            trainable_branches = [branch for branch in rollout.raw.branches if any(branch.sampled_mask)]
            if len(trainable_branches) != len(rollout.samples):
                raise ValueError(
                    f"sdpo expected one sample per trainable branch; got {len(rollout.samples)} sample(s) "
                    f"for {len(trainable_branches)} branch(es) (env '{rollout.env_name}')."
                )
            for branch, sample in zip(trainable_branches, rollout.samples):
                self._validate_branch_sample_alignment(branch, sample, rollout.env_name)
                student_support_token_rows = None
                if self.distillation_topk_support == "student":
                    student_support_token_rows = sample.sdpo_topk_token_ids
                self._ensure_sdpo_weights(sample)
                output_support_initialized = False

                def ensure_output_support_initialized() -> None:
                    nonlocal output_support_initialized
                    if output_support_initialized:
                        return
                    sample.sdpo_topk_token_ids = [[0] * self.distillation_topk for _ in sample.token_ids]
                    sample.sdpo_topk_logprobs = [[0.0] * self.distillation_topk for _ in sample.token_ids]
                    output_support_initialized = True

                offset = 0
                for node_index, node in enumerate(branch.nodes):
                    span = range(offset, offset + len(node.token_ids))
                    positions = self._sdpo_target_positions(sample, [i for i in span if sample.mask[i]])
                    if node.sampled and positions:
                        messages = [prior.message.model_dump(exclude_none=True) for prior in branch.nodes[:node_index]]
                        raw_feedback = self._feedback_after_node(branch, node_index)
                        feedback = raw_feedback if self._use_feedback(solution=solution, feedback=raw_feedback) else ""
                        if solution is None and not feedback:
                            self._zero_sdpo_positions(sample, positions)
                            offset += len(node.token_ids)
                            continue
                        self._reject_precomputed_rollout_is_weights(sample, positions)
                        ensure_output_support_initialized()
                        prefix_ids = self._teacher_prefix_ids(
                            messages,
                            successful_previous_rollout=solution,
                            hindsight_feedback=feedback,
                            env_name=rollout.env_name,
                        )
                        target_offsets = [i - offset for i in positions]
                        completion_ids = list(node.token_ids)
                        if self.distillation_topk_support == "student":
                            token_rows = candidate_rows_from_sample(
                                sample,
                                positions,
                                rollout.env_name,
                                token_rows=student_support_token_rows,
                            )
                            logprob_rows = await score_student_support_completion(
                                client, prefix_ids, completion_ids, target_offsets, token_rows
                            )
                        else:
                            token_rows, logprob_rows = await score_teacher_topk_completion(
                                client, prefix_ids, completion_ids, target_offsets
                            )
                        self._validate_topk_rows(token_rows, logprob_rows, len(positions), rollout.env_name)
                        for i, token_row, logprob_row in zip(positions, token_rows, logprob_rows):
                            sample.sdpo_topk_token_ids[i] = token_row
                            sample.sdpo_topk_logprobs[i] = logprob_row
                    offset += len(node.token_ids)
                if not self._has_sdpo_weight(sample):
                    self._clear_sdpo_target(sample)

        async def score_one(client, rollout: RolloutView) -> None:
            if rollout.raw.num_turns == 1:
                await score_single_turn(client, rollout)
            elif self.multi_turn:
                await score_multi_turn(client, rollout)
            else:
                raise ValueError(
                    f"sdpo supports single-step trajectories by default; "
                    f"env '{rollout.env_name}' produced {rollout.raw.num_turns} model turn(s)."
                )

        await asyncio.gather(*[score_one(client, rollout) for client, rollout in zip(cycle(train_clients), batch)])

    def _scatter_topk(
        self,
        sample: TrainingSample,
        token_rows: list[list[int]],
        logprob_rows: list[list[float]],
        *,
        positions: list[int] | None = None,
    ) -> None:
        target_positions = self._sdpo_target_positions(sample)
        if positions is not None:
            if positions != target_positions:
                raise ValueError(
                    f"sdpo scatter target positions changed before writeback "
                    f"(env '{sample.env_name}', scored={positions}, current={target_positions})."
                )
            target_positions = positions
        self._validate_topk_rows(token_rows, logprob_rows, len(target_positions), sample.env_name)
        sample.sdpo_topk_token_ids = [[0] * self.distillation_topk for _ in sample.token_ids]
        sample.sdpo_topk_logprobs = [[0.0] * self.distillation_topk for _ in sample.token_ids]
        for i, token_row, logprob_row in zip(target_positions, token_rows, logprob_rows):
            sample.sdpo_topk_token_ids[i] = token_row
            sample.sdpo_topk_logprobs[i] = logprob_row

    def _validate_topk_rows(
        self,
        token_rows: list[list[int]],
        logprob_rows: list[list[float]],
        expected_count: int,
        env_name: str,
    ) -> None:
        if len(token_rows) != expected_count or len(logprob_rows) != expected_count:
            raise ValueError(
                f"sdpo teacher top-k row count mismatch (env '{env_name}', "
                f"expected={expected_count}, token_rows={len(token_rows)}, logprob_rows={len(logprob_rows)})."
            )
        for row_idx, (token_row, logprob_row) in enumerate(zip(token_rows, logprob_rows)):
            if not isinstance(token_row, list):
                raise ValueError(
                    f"sdpo teacher top-k token row must be a list "
                    f"(env '{env_name}', row={row_idx}, got={type(token_row).__name__})."
                )
            if not isinstance(logprob_row, list):
                raise ValueError(
                    f"sdpo teacher top-k logprob row must be a list "
                    f"(env '{env_name}', row={row_idx}, got={type(logprob_row).__name__})."
                )
            if len(token_row) != self.distillation_topk or len(logprob_row) != self.distillation_topk:
                raise ValueError(
                    f"sdpo teacher top-k row width mismatch (env '{env_name}', row={row_idx}, "
                    f"expected={self.distillation_topk}, token_row={len(token_row)}, "
                    f"logprob_row={len(logprob_row)})."
                )
            if not all(not isinstance(token_id, bool) and isinstance(token_id, int) for token_id in token_row):
                raise ValueError(
                    f"sdpo teacher top-k row contains non-integer token ids (env '{env_name}', row={row_idx})."
                )
            if not all(token_id >= 0 for token_id in token_row):
                raise ValueError(
                    f"sdpo teacher top-k row contains negative token ids (env '{env_name}', row={row_idx})."
                )
            if len(set(token_row)) != len(token_row):
                raise ValueError(
                    f"sdpo teacher top-k row contains duplicate token ids (env '{env_name}', row={row_idx})."
                )
            if not all(isinstance(logprob, float) and math.isfinite(logprob) for logprob in logprob_row):
                raise ValueError(
                    f"sdpo teacher top-k row must contain floating-point logprobs (env '{env_name}', row={row_idx})."
                )
            if _is_placeholder_logprob_row(logprob_row):
                raise ValueError(
                    f"sdpo teacher top-k row contains placeholder logprobs (env '{env_name}', row={row_idx})."
                )
            row_mass = math.fsum(math.exp(float(logprob)) for logprob in logprob_row)
            if row_mass > 1.0 + 1e-5:
                raise ValueError(
                    f"sdpo teacher top-k row probability mass exceeds 1 "
                    f"(env '{env_name}', row={row_idx}, mass={row_mass:.6g})."
                )

    def _validate_token_rows(
        self,
        token_rows: list[list[int]],
        expected_count: int,
        env_name: str,
    ) -> None:
        if len(token_rows) != expected_count:
            raise ValueError(
                f"sdpo student top-k row count mismatch (env '{env_name}', "
                f"expected={expected_count}, token_rows={len(token_rows)})."
            )
        for row_idx, token_row in enumerate(token_rows):
            if not isinstance(token_row, list):
                raise ValueError(
                    f"sdpo student top-k row must be a list "
                    f"(env '{env_name}', row={row_idx}, got={type(token_row).__name__})."
                )
            if len(token_row) != self.distillation_topk:
                raise ValueError(
                    f"sdpo student top-k row width mismatch (env '{env_name}', row={row_idx}, "
                    f"expected={self.distillation_topk}, token_row={len(token_row)})."
                )
            if not all(not isinstance(token_id, bool) and isinstance(token_id, int) for token_id in token_row):
                raise ValueError(
                    f"sdpo student top-k row contains non-integer token ids (env '{env_name}', row={row_idx})."
                )
            if not all(token_id >= 0 for token_id in token_row):
                raise ValueError(
                    f"sdpo student top-k row contains negative token ids (env '{env_name}', row={row_idx})."
                )
            if len(set(token_row)) != len(token_row):
                raise ValueError(
                    f"sdpo student top-k row contains duplicate token ids (env '{env_name}', row={row_idx})."
                )

    def _student_support_chunks(self, token_rows: list[list[int]]) -> list[tuple[int, int]]:
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < len(token_rows):
            candidate_union: set[int] = set()
            end = start
            while end < len(token_rows):
                next_union = candidate_union | set(token_rows[end])
                if len(next_union) > MAX_PREFILL_CANDIDATE_TOKEN_IDS:
                    if end == start:
                        raise ValueError(
                            "sdpo student top-k row has "
                            f"{len(next_union)} unique token ids, exceeding vLLM's "
                            f"{MAX_PREFILL_CANDIDATE_TOKEN_IDS} candidate-token limit."
                        )
                    break
                candidate_union = next_union
                end += 1
            chunks.append((start, end))
            start = end
        return chunks
