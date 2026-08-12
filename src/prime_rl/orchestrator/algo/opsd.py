from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Callable

from verifiers.v1.types import content_text

from prime_rl.configs.algorithm import OPSDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.transport import SDPOTeacherSpan
from prime_rl.utils.utils import import_object

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
        self.filter_fn: Callable[..., list[list[bool]]] | None = None
        if config.filter is not None:
            self.filter_fn = partial(import_object(config.filter.import_path), **config.filter.kwargs)

    async def setup(self) -> None:
        from renderers.base import create_renderer, load_tokenizer

        tokenizer = load_tokenizer(self.policy_pool.model_name)
        self.renderer = create_renderer(tokenizer, self.config.renderer)

    async def score_rollout(self, rollout: Rollout) -> None:
        demonstrations = self._demonstrations(rollout)
        trainable_branches = list(iter_trainable_branches(rollout))
        if len(trainable_branches) != len(rollout.samples):
            raise ValueError("OPSD samples must align one-to-one with trainable trace branches")
        filter_masks = (
            self._filter_masks(rollout, [branch for branch, _ in trainable_branches])
            if self.filter_fn is not None
            else None
        )

        for branch_index, ((branch, trainable_mask), sample) in enumerate(
            zip(trainable_branches, rollout.samples, strict=True)
        ):
            if list(branch.token_ids) != list(sample.token_ids) or list(trainable_mask) != list(sample.mask):
                raise ValueError("OPSD sample tokens must align with their Verifiers trace branch")
            demonstration = self._branch_demonstration(rollout, branch.nodes, demonstrations)
            if demonstration is None:
                sample.sdpo_weights = [0.0] * len(sample.token_ids)
                sample.sdpo_teacher_spans = None
                continue
            keep_mask = filter_masks[branch_index] if filter_masks is not None else None
            self._prepare_sample(rollout, branch.nodes, sample, demonstration, keep_mask)

    def _prepare_sample(
        self,
        rollout: Rollout,
        nodes: list,
        sample: TrainingSample,
        demonstration: str | Sequence[str | None],
        keep_mask: list[bool] | None = None,
    ) -> None:
        weights = [0.0] * len(sample.token_ids)
        spans: list[SDPOTeacherSpan] = []
        demonstration_index = 0
        for node_index, sampled_node in enumerate(nodes):
            if not sampled_node.sampled:
                continue
            node_start = sum(len(node.token_ids) for node in nodes[:node_index])
            node_end = node_start + len(sampled_node.token_ids)
            action_mask = sample.mask[node_start:node_end]
            if len(action_mask) != len(sampled_node.token_ids):
                raise ValueError("OPSD sample mask must span every Verifiers trace node")
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
                raise ValueError("OPSD sampled response tokens must be present and position-aligned")

            node_demonstration = (
                demonstration
                if isinstance(demonstration, str)
                else demonstration[demonstration_index]
                if demonstration_index < len(demonstration)
                else None
            )
            demonstration_index += 1
            if node_demonstration is None:
                continue

            messages = [node.message for node in nodes[:node_index]]
            prefix_ids = self._teacher_prefix_ids(messages, rollout.tools, node_demonstration)
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

        sample.sdpo_weights = weights
        sample.sdpo_teacher_spans = spans or None

    def _filter_masks(self, rollout: Rollout, trainable_branches: list) -> list[list[bool]]:
        assert self.filter_fn is not None
        masks = self.filter_fn(rollout)
        if not isinstance(masks, list) or len(masks) != len(trainable_branches):
            got = len(masks) if isinstance(masks, list) else type(masks).__name__
            raise ValueError(
                f"opsd filter must return one keep-mask per trainable branch: got {got}, "
                f"expected {len(trainable_branches)}"
            )
        for branch_index, (branch, mask) in enumerate(zip(trainable_branches, masks, strict=True)):
            expected = len(branch.token_ids)
            if not isinstance(mask, list) or len(mask) != expected:
                got = len(mask) if isinstance(mask, list) else type(mask).__name__
                raise ValueError(
                    f"opsd filter mask for branch {branch_index} must span the branch's tokens: "
                    f"got {got}, expected {expected}"
                )
        return masks

    def _teacher_prefix_ids(self, messages: list, tools: list, demonstration: str) -> list[int]:
        renderer = self.renderer
        if renderer is None:
            raise RuntimeError("OPSD renderer is not initialized; call Algorithm.setup() first")
        user_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].role == "user"),
            None,
        )
        if user_index is None:
            raise ValueError("OPSD teacher context requires a user message")
        question = content_text(messages[user_index].content).strip()
        if not question:
            raise ValueError("OPSD teacher context requires a text question")
        rendered_messages = [message.model_dump(exclude_none=True) for message in messages]
        rendered_messages[user_index]["content"] = self.config.template.format(
            question=question,
            demonstration=demonstration,
        )
        rendered_tools = [tool.model_dump(exclude_none=True) for tool in tools]
        prefix_ids = list(renderer.render_ids(rendered_messages, tools=rendered_tools, add_generation_prompt=True))
        return prefix_ids[: self.config.max_reprompt_len]

    def _demonstrations(
        self,
        rollout: Rollout,
    ) -> str | Mapping[str, str | Sequence[str | None] | None]:
        demonstrations = rollout.info.get(self.config.demo_key)
        if demonstrations is None:
            demonstrations = getattr(rollout.task.data, self.config.demo_key, None)
        if not isinstance(demonstrations, (str, Mapping)):
            raise ValueError(
                f"opsd requires string or question-keyed mapping '{self.config.demo_key}' in the trace info dict "
                f"or on the task "
                f"(env '{rollout.env_name}', task {rollout.task.data.idx})."
            )
        def valid_demonstration(value: object) -> bool:
            return (
                isinstance(value, str)
                or value is None
                or (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes))
                    and all(isinstance(item, str) or item is None for item in value)
                )
            )

        if isinstance(demonstrations, Mapping) and not all(
            isinstance(question, str) and valid_demonstration(demonstration)
            for question, demonstration in demonstrations.items()
        ):
            raise ValueError(
                f"opsd demonstration mapping '{self.config.demo_key}' must map strings to "
                "strings, string-or-null sequences, or null"
            )
        return demonstrations

    def _branch_demonstration(
        self,
        rollout: Rollout,
        nodes: list,
        demonstrations: str | Mapping[str, str | Sequence[str | None] | None],
    ) -> str | Sequence[str | None] | None:
        if isinstance(demonstrations, str):
            return demonstrations
        question = next(
            (
                content_text(node.message.content).strip()
                for node in nodes
                if node.message.role == "user" and content_text(node.message.content).strip()
            ),
            None,
        )
        if question is None:
            raise ValueError("OPSD branch-specific demonstration requires an initial user question")
        missing = object()
        demonstration = demonstrations.get(question, missing)
        if demonstration is missing:
            demonstration = demonstrations.get("*", missing)
        if demonstration is missing:
            raise ValueError(
                f"opsd found no demonstration for branch question {question!r} "
                f"(env '{rollout.env_name}', task {rollout.task.data.idx})"
            )
        assert isinstance(demonstration, (str, Sequence)) or demonstration is None
        return demonstration
