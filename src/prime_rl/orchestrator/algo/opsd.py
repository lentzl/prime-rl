from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import OPSDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.transport import SDPOTeacherSpan

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import Rollout
    from prime_rl.transport import TrainingSample
    from prime_rl.utils.client import InferencePool


class OPSDAlgorithm(Algorithm):
    """On-policy self-distillation from per-example expert demonstrations."""

    action_loss_type = "sdpo"

    def __init__(self, config: OPSDAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.config = config
        self.renderer: Renderer | None = None

    async def setup(self) -> None:
        from renderers.base import create_renderer, load_tokenizer

        tokenizer = load_tokenizer(self.policy_pool.model_name)
        self.renderer = create_renderer(tokenizer, self.config.renderer)

    async def score_rollout(self, rollout: Rollout) -> None:
        demonstration = self._demonstration(rollout)
        trainable_branches = [branch for branch in rollout.branches if any(branch.sampled_mask)]
        if len(trainable_branches) != len(rollout.samples):
            raise ValueError("OPSD samples must align one-to-one with trainable trace branches")

        for branch, sample in zip(trainable_branches, rollout.samples, strict=True):
            if list(branch.token_ids) != list(sample.token_ids) or list(branch.sampled_mask) != list(sample.mask):
                raise ValueError("OPSD sample tokens must align with their Verifiers trace branch")
            self._prepare_sample(rollout, branch.nodes, sample, demonstration)

    def _prepare_sample(self, rollout: Rollout, nodes: list, sample: TrainingSample, demonstration: str) -> None:
        weights = [0.0] * len(sample.token_ids)
        spans: list[SDPOTeacherSpan] = []
        for node_index, sampled_node in enumerate(nodes):
            if not sampled_node.sampled:
                continue
            node_start = sum(len(node.token_ids) for node in nodes[:node_index])
            student_positions = [node_start + index for index, sampled in enumerate(sampled_node.mask) if sampled]
            completion_ids = [
                token_id for token_id, sampled in zip(sampled_node.token_ids, sampled_node.mask, strict=True) if sampled
            ]
            if not completion_ids or len(student_positions) != len(completion_ids):
                raise ValueError("OPSD sampled response tokens must be present and position-aligned")

            messages = [node.message.model_dump(exclude_none=True) for node in nodes[:node_index]]
            prefix_ids = self._teacher_prefix_ids(messages, rollout.tools, demonstration)
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

        sample.sdpo_weights = weights
        sample.sdpo_teacher_spans = spans or None

    def _teacher_prefix_ids(self, messages: list[dict], tools: list, demonstration: str) -> list[int]:
        renderer = self.renderer
        if renderer is None:
            raise RuntimeError("OPSD renderer is not initialized; call Algorithm.setup() first")
        messages = [dict(message) for message in messages]
        user_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"),
            None,
        )
        if user_index is None:
            raise ValueError("OPSD teacher context requires a user message")
        question = messages[user_index].get("content")
        if not isinstance(question, str):
            raise ValueError("OPSD currently supports text-only prompts")
        messages[user_index]["content"] = self.config.template.format(
            question=question,
            demonstration=demonstration,
        )
        rendered_tools = [tool.model_dump(exclude_none=True) for tool in tools]
        prefix_ids = list(renderer.render_ids(messages, tools=rendered_tools, add_generation_prompt=True))
        return prefix_ids[: self.config.max_reprompt_len]

    def _demonstration(self, rollout: Rollout) -> str:
        demonstration = rollout.info.get(self.config.demo_key)
        if demonstration is None:
            demonstration = getattr(rollout.task.data, self.config.demo_key, None)
        if not isinstance(demonstration, str):
            raise ValueError(
                f"opsd requires string '{self.config.demo_key}' in the trace info dict or on the task "
                f"(env '{rollout.env_name}', task {rollout.task.data.idx})."
            )
        return demonstration
