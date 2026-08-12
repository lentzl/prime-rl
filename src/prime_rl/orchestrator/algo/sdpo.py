from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from verifiers.v1.types import content_text, content_to_parts

from prime_rl.configs.algorithm import SDPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.transport import SDPOTeacherSpan
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import Rollout
    from prime_rl.transport import TrainingSample
    from prime_rl.utils.client import InferencePool


class SDPOAlgorithm(Algorithm):
    """Feedback-conditioned self-distillation over the policy's own attempts."""

    action_loss_type = "sdpo"

    def __init__(self, config: SDPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.config = config
        self.renderer: Renderer | None = None
        self.filter_fn: Callable[..., list[list[bool]]] | None = None
        if config.filter is not None:
            self.filter_fn = partial(import_object(config.filter.import_path), **config.filter.kwargs)

    async def setup(self) -> None:
        from renderers.base import create_renderer, load_tokenizer

        tokenizer = load_tokenizer(self.policy_pool.model_name)
        self.renderer = create_renderer(tokenizer, self.config.renderer)

    async def score_group(self, group: list[Rollout]) -> None:
        demonstrations = [
            (rollout, self._branch_demonstrations(rollout))
            for rollout in group
            if rollout.reward >= self.config.success_reward_threshold
        ]
        for rollout in group:
            solutions = next(
                (
                    branch_demonstrations
                    for candidate, branch_demonstrations in demonstrations
                    if not self.config.dont_reprompt_on_self_success or candidate is not rollout
                ),
                None,
            )
            self._prepare_rollout(rollout, solutions)

    def _prepare_rollout(self, rollout: Rollout, solutions: dict[str, str] | None) -> None:
        trainable_branches = list(iter_trainable_branches(rollout))
        if len(trainable_branches) != len(rollout.samples):
            raise ValueError("SDPO samples must align one-to-one with trainable trace branches")
        filter_masks = (
            self._filter_masks(rollout, [branch for branch, _ in trainable_branches])
            if self.filter_fn is not None
            else None
        )
        explicit_feedback = rollout.info.get("feedback")
        if not isinstance(explicit_feedback, str):
            explicit_feedback = None

        for branch_index, ((branch, trainable_mask), sample) in enumerate(
            zip(trainable_branches, rollout.samples, strict=True)
        ):
            if list(branch.token_ids) != list(sample.token_ids) or list(trainable_mask) != list(sample.mask):
                raise ValueError("SDPO sample tokens must align with their Verifiers trace branch")
            keep_mask = filter_masks[branch_index] if filter_masks is not None else None
            if keep_mask is not None and not any(
                sampled and keep for sampled, keep in zip(sample.mask, keep_mask, strict=True)
            ):
                self._clear_target(sample)
                continue
            question = self._branch_question(branch.nodes)
            if solutions is None:
                solution = None
            else:
                if question not in solutions:
                    raise ValueError(
                        f"SDPO found no successful sibling branch for question {question!r} "
                        f"(env '{rollout.env_name}', task {rollout.task.data.idx})"
                    )
                solution = solutions[question]
            self._prepare_sample(
                rollout,
                branch.nodes,
                sample,
                solution,
                explicit_feedback=explicit_feedback if branch_index == 0 else None,
                keep_mask=keep_mask,
            )

    def _prepare_sample(
        self,
        rollout: Rollout,
        nodes: list,
        sample: TrainingSample,
        solution: str | None,
        *,
        explicit_feedback: str | None,
        keep_mask: list[bool] | None = None,
    ) -> None:

        sampled_nodes: list[tuple[int, object, int, int]] = []
        node_start = 0
        for node_index, node in enumerate(nodes):
            node_end = node_start + len(node.token_ids)
            if node.sampled:
                action_mask = sample.mask[node_start:node_end]
                node_keep_mask = keep_mask[node_start:node_end] if keep_mask is not None else action_mask
                if any(
                    sampled and keep
                    for sampled, keep in zip(action_mask, node_keep_mask, strict=True)
                ):
                    sampled_nodes.append((node_index, node, node_start, node_end))
            node_start = node_end
        if len(sampled_nodes) != 1 and not self.config.multi_turn_replay:
            raise ValueError("SDPO currently requires single-turn rollouts")

        if rollout.reward >= self.config.success_reward_threshold and solution is None:
            self._clear_target(sample)
            return

        weights = [0.0] * len(sample.token_ids)
        spans: list[SDPOTeacherSpan] = []
        for turn_index, (node_index, sampled_node, node_start, node_end) in enumerate(sampled_nodes):
            action_mask = sample.mask[node_start:node_end]
            if len(action_mask) != len(sampled_node.token_ids):
                raise ValueError("SDPO sample mask must span every Verifiers trace node")
            effective_mask = action_mask
            if keep_mask is not None:
                node_keep_mask = keep_mask[node_start:node_end]
                effective_mask = [
                    sampled and keep
                    for sampled, keep in zip(action_mask, node_keep_mask, strict=True)
                ]
            student_positions = [node_start + index for index, sampled in enumerate(effective_mask) if sampled]
            target_offsets: list[int] = []
            completion_ids: list[int] = []
            for token_id, is_action, is_selected in zip(
                sampled_node.token_ids,
                action_mask,
                effective_mask,
                strict=True,
            ):
                if not is_action:
                    continue
                if is_selected:
                    target_offsets.append(len(completion_ids))
                completion_ids.append(token_id)
            if not student_positions:
                continue
            if len(student_positions) != len(target_offsets):
                raise ValueError("SDPO sampled response tokens must be present and position-aligned")

            feedback = self._turn_feedback(
                nodes,
                node_index,
                is_final_turn=turn_index == len(sampled_nodes) - 1,
                explicit_feedback=explicit_feedback,
            )
            if solution is not None and self.config.environment_feedback_only_without_solution:
                feedback = ""
            if not self.config.include_environment_feedback:
                feedback = ""
            if solution is None and not feedback:
                continue

            messages = [node.message.model_dump(exclude_none=True) for node in nodes[:node_index]]
            prefix_ids = self._teacher_prefix_ids(messages, solution=solution, feedback=feedback)
            for position in student_positions:
                weights[position] = 1.0
            spans.append(
                SDPOTeacherSpan(
                    prefix_ids=prefix_ids,
                    completion_ids=completion_ids,
                    student_positions=student_positions,
                    target_offsets=target_offsets,
                )
            )

        if not spans:
            self._clear_target(sample)
            return
        sample.sdpo_weights = weights
        sample.sdpo_teacher_spans = spans

    def _filter_masks(self, rollout: Rollout, trainable_branches: list) -> list[list[bool]]:
        assert self.filter_fn is not None
        masks = self.filter_fn(rollout)
        if not isinstance(masks, list) or len(masks) != len(trainable_branches):
            got = len(masks) if isinstance(masks, list) else type(masks).__name__
            raise ValueError(
                f"sdpo filter must return one keep-mask per trainable branch: got {got}, "
                f"expected {len(trainable_branches)}"
            )
        for branch_index, (branch, mask) in enumerate(zip(trainable_branches, masks, strict=True)):
            expected = len(branch.token_ids)
            if not isinstance(mask, list) or len(mask) != expected:
                got = len(mask) if isinstance(mask, list) else type(mask).__name__
                raise ValueError(
                    f"sdpo filter mask for branch {branch_index} must span the branch's tokens: "
                    f"got {got}, expected {expected}"
                )
        return masks

    def _teacher_prefix_ids(self, messages: list[dict], *, solution: str | None, feedback: str) -> list[int]:
        renderer = self.renderer
        if renderer is None:
            raise RuntimeError("SDPO renderer is not initialized; call Algorithm.setup() first")
        messages = [dict(message) for message in messages]
        user_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"),
            None,
        )
        if user_index is None:
            raise ValueError("SDPO teacher reprompt requires a user message")
        question = content_text(content_to_parts(messages[user_index].get("content"))).strip()
        if not question:
            raise ValueError("SDPO teacher reprompt requires a text question")
        solution_block = (
            self.config.solution_template.format(successful_previous_attempt=solution) if solution is not None else ""
        )
        feedback_block = self.config.feedback_template.format(feedback_raw=feedback) if feedback else ""
        messages[user_index]["content"] = self.config.template.format(
            question=question,
            successful_solution_block=solution_block,
            feedback_block=feedback_block,
        )
        prefix_ids = list(renderer.render_ids(messages, add_generation_prompt=True))
        return prefix_ids[: self.config.max_reprompt_len]

    def _turn_feedback(
        self,
        nodes: list,
        sampled_node_index: int,
        *,
        is_final_turn: bool,
        explicit_feedback: str | None,
    ) -> str:
        if is_final_turn and not self.config.multi_turn_replay and explicit_feedback and explicit_feedback.strip():
            return explicit_feedback
        feedback: list[str] = []
        for node in nodes[sampled_node_index + 1 :]:
            if node.sampled:
                break
            text = content_text(node.message.content).strip()
            if text:
                feedback.append(f"{node.message.role}: {text}")
        if feedback:
            return "\n".join(feedback)
        if is_final_turn and explicit_feedback and explicit_feedback.strip():
            return explicit_feedback
        return ""

    def _branch_demonstrations(self, rollout: Rollout) -> dict[str, str]:
        demonstrations: dict[str, str] = {}
        for branch, _ in iter_trainable_branches(rollout):
            question = self._branch_question(branch.nodes)
            if question in demonstrations:
                raise ValueError(
                    f"SDPO cannot match duplicate trainable branch question {question!r} "
                    f"(env '{rollout.env_name}', task {rollout.task.data.idx})"
                )
            demonstrations[question] = self._branch_demonstration(branch.nodes)
        return demonstrations

    def _branch_demonstration(self, nodes: list) -> str:
        turns: list[str] = []
        for node in nodes:
            message = node.message
            if message.role != "assistant":
                continue
            reasoning = getattr(message, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning.strip():
                turns.append(reasoning)
            for tool_call in getattr(message, "tool_calls", None) or []:
                turns.append(f"{tool_call.name}({tool_call.arguments})")
            text = content_text(message.content)
            if text.strip():
                turns.append(text)
        text = "\n".join(turns)
        if self.config.remove_thinking_from_demonstration:
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        return text

    @staticmethod
    def _branch_question(nodes: list) -> str:
        question = next(
            (
                content_text(node.message.content).strip()
                for node in nodes
                if node.message.role == "user" and content_text(node.message.content).strip()
            ),
            None,
        )
        if question is None:
            raise ValueError("SDPO branch matching requires an initial user question")
        return question

    @staticmethod
    def _clear_target(sample: TrainingSample) -> None:
        sample.sdpo_weights = [0.0] * len(sample.token_ids)
        sample.sdpo_teacher_spans = None
