from __future__ import annotations

import re
from typing import TYPE_CHECKING

from verifiers.v1.types import content_text

from prime_rl.configs.algorithm import SDPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.transport import SDPOTeacherSpan

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

    async def setup(self) -> None:
        from renderers.base import create_renderer, load_tokenizer

        tokenizer = load_tokenizer(self.policy_pool.model_name)
        self.renderer = create_renderer(tokenizer, self.config.renderer)

    async def score_group(self, group: list[Rollout]) -> None:
        demonstrations = [
            (rollout, self._demonstration(rollout))
            for rollout in group
            if rollout.reward >= self.config.success_reward_threshold
        ]
        for rollout in group:
            solution = next(
                (
                    text
                    for candidate, text in demonstrations
                    if not self.config.dont_reprompt_on_self_success or candidate is not rollout
                ),
                None,
            )
            self._prepare_rollout(rollout, solution)

    def _prepare_rollout(self, rollout: Rollout, solution: str | None) -> None:
        trainable_branches = [branch for branch in rollout.branches if any(branch.sampled_mask)]
        if len(trainable_branches) != 1 or len(rollout.samples) != 1:
            raise ValueError("SDPO currently requires one trainable branch per rollout")
        branch = trainable_branches[0]
        sample = rollout.samples[0]
        if list(branch.token_ids) != list(sample.token_ids) or list(branch.sampled_mask) != list(sample.mask):
            raise ValueError("SDPO sample tokens must align with their Verifiers trace branch")

        explicit_feedback = self._explicit_feedback(rollout)
        if (
            self.config.require_explicit_feedback or self.config.required_feedback_contract_schema is not None
        ) and explicit_feedback is None:
            self._clear_target(sample)
            return

        sampled_nodes = [(index, node) for index, node in enumerate(branch.nodes) if node.sampled]
        if len(sampled_nodes) != 1 and not self.config.multi_turn_replay:
            raise ValueError("SDPO currently requires single-turn rollouts")

        if rollout.reward >= self.config.success_reward_threshold and solution is None:
            self._clear_target(sample)
            return

        weights = [0.0] * len(sample.token_ids)
        spans: list[SDPOTeacherSpan] = []
        for turn_index, (node_index, sampled_node) in enumerate(sampled_nodes):
            feedback = self._turn_feedback(
                rollout,
                branch.nodes,
                node_index,
                is_final_turn=turn_index == len(sampled_nodes) - 1,
            )
            if solution is not None and self.config.environment_feedback_only_without_solution:
                feedback = ""
            if not self.config.include_environment_feedback:
                feedback = ""
            if solution is None and not feedback:
                continue

            node_start = sum(len(node.token_ids) for node in branch.nodes[:node_index])
            student_positions = [node_start + index for index, sampled in enumerate(sampled_node.mask) if sampled]
            completion_ids = [
                token_id for token_id, sampled in zip(sampled_node.token_ids, sampled_node.mask) if sampled
            ]
            if not completion_ids or len(student_positions) != len(completion_ids):
                raise ValueError("SDPO sampled response tokens must be present and position-aligned")

            messages = [node.message.model_dump(exclude_none=True) for node in branch.nodes[:node_index]]
            prefix_ids = self._teacher_prefix_ids(messages, solution=solution, feedback=feedback)
            for position in student_positions:
                weights[position] = 1.0
            spans.append(
                SDPOTeacherSpan(
                    prefix_ids=prefix_ids,
                    completion_ids=completion_ids,
                    student_positions=student_positions,
                    target_offsets=list(range(len(completion_ids))),
                )
            )

        if not spans:
            self._clear_target(sample)
            return
        sample.sdpo_weights = weights
        sample.sdpo_teacher_spans = spans

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
        question = messages[user_index].get("content")
        if isinstance(question, list) and all(
            isinstance(part, dict) and part.get("type") == "text" for part in question
        ):
            question = "\n".join(str(part.get("text", "")) for part in question)
        if not isinstance(question, str):
            raise ValueError("SDPO currently supports text-only prompts")
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

    def _turn_feedback(self, rollout: Rollout, nodes: list, sampled_node_index: int, *, is_final_turn: bool) -> str:
        explicit = self._explicit_feedback(rollout)
        if is_final_turn and not self.config.multi_turn_replay and explicit is not None:
            return explicit
        feedback: list[str] = []
        for node in nodes[sampled_node_index + 1 :]:
            if node.sampled:
                break
            text = content_text(node.message.content).strip()
            if text:
                feedback.append(f"{node.message.role}: {text}")
        if feedback:
            return "\n".join(feedback)
        if is_final_turn and explicit is not None:
            return explicit
        return ""

    def _explicit_feedback(self, rollout: Rollout) -> str | None:
        explicit = rollout.info.get("feedback")
        if not isinstance(explicit, str) or not explicit.strip():
            return None
        required_schema = self.config.required_feedback_contract_schema
        if required_schema is None:
            return explicit
        contract = rollout.info.get("feedback_contract")
        if not isinstance(contract, dict):
            return None
        if contract.get("schema_version") != required_schema:
            return None
        if contract.get("answer_free") is not True or contract.get("retryable") is not True:
            return None
        if not all(isinstance(contract.get(key), str) and contract[key].strip() for key in ("code", "category")):
            return None
        if contract.get("message") != explicit:
            return None
        return explicit

    def _demonstration(self, rollout: Rollout) -> str:
        text = "\n".join(content_text(message.content) for message in rollout.assistant_messages)
        if self.config.remove_thinking_from_demonstration:
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        return text

    @staticmethod
    def _clear_target(sample: TrainingSample) -> None:
        sample.sdpo_weights = [0.0] * len(sample.token_ids)
        sample.sdpo_teacher_spans = None
