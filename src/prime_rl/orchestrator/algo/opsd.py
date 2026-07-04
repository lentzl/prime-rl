from __future__ import annotations

import asyncio
from collections.abc import Mapping
from itertools import cycle
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import AlgorithmConfig, OPSDAlgorithmConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.utils import compute_prefill_logprobs

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import RolloutView
    from prime_rl.utils.client import InferencePool


class OPSDAlgorithm(Algorithm):
    """On-policy self-distillation (SDFT). The teacher defaults to the policy
    itself, conditioned on an expert demonstration — no extra deployment.

    The scoring prefix is rebuilt from the rollout's first-turn prompt
    messages with the demonstration woven into the last user message; the
    returned completion logprobs are aligned back onto the sample (the
    sample's prompt positions are 0.0 and stay outside the loss mask). No
    scalar advantage is assigned."""

    action_loss_type = "ref_kl"
    model_role = "teacher"

    def __init__(
        self,
        config: AlgorithmConfig,
        policy_pool: InferencePool,
        renderer: Renderer | None,
        live_pools: Mapping[str, InferencePool] | None = None,
    ):
        super().__init__(config, policy_pool, renderer, live_pools=live_pools)
        assert isinstance(config, OPSDAlgorithmConfig)
        assert renderer is not None, "opsd requires the renderer (validated at config time)"
        self.demo_key = config.demo_key
        self.template = config.template
        self.template_target = config.template_target
        self.max_concurrent = config.max_concurrent
        self.multi_turn = config.multi_turn
        self.teacher = config.model
        self.teacher_pool: InferencePool | None = None  # connected in setup()

    async def setup(self) -> None:
        self.teacher_pool = await self.connect(self.teacher)

    def _demonstration(self, rollout: RolloutView) -> str:
        trace = rollout.raw
        demonstration = trace.info.get(self.demo_key)
        if demonstration is None:
            demonstration = getattr(trace.task, self.demo_key, None)
        if demonstration is None:
            raise ValueError(
                f"opsd requires '{self.demo_key}' in the trace info dict or on the task "
                f"(env '{rollout.env_name}', task {trace.task.idx})."
            )
        return demonstration

    def _feedback_after_node(self, branch, node_index: int) -> str:
        feedback: list[str] = []
        for node in branch.nodes[node_index + 1 :]:
            if node.sampled:
                break
            message = node.message.model_dump(exclude_none=True)
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                role = message.get("role", "message")
                feedback.append(f"{role}: {content}")
        return "\n".join(feedback)

    def _template_fields(
        self,
        rollout: RolloutView,
        demonstration: str,
        branch=None,
        node_index: int | None = None,
    ) -> dict[str, str]:
        fields = {
            key: str(value)
            for key, value in rollout.raw.info.items()
            if key not in {"question", "demonstration"} and isinstance(value, str | int | float | bool)
        } | {"demonstration": demonstration, "hindsight_feedback": ""}
        if branch is not None and node_index is not None:
            fields["hindsight_feedback"] = self._feedback_after_node(branch, node_index)
        return fields

    def _demo_conditioned_prefix_ids(
        self,
        messages: list[dict],
        demonstration: str,
        env_name: str,
        *,
        template_fields: dict[str, str] | None = None,
    ) -> list[int]:
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            raise ValueError(f"opsd found no user message to condition (env '{env_name}').")
        user_index = user_indices[0] if self.template_target == "first_user" else user_indices[-1]
        user_message = messages[user_index]
        question = user_message.get("content")
        if not isinstance(question, str):
            raise ValueError("opsd supports text-only prompts (user content must be a string).")
        fields = {"question": question, **(template_fields or {}), "demonstration": demonstration}
        user_message["content"] = self.template.format(**fields)

        # Render through the policy's renderer — the same messages → token ids
        # path the policy's own prompts take, so the scoring prefix matches
        # the prompt distribution the teacher conditions on.
        assert self.renderer is not None
        return self.renderer.render_ids(messages, add_generation_prompt=True)

    def _ref_prefix_ids(self, rollout: RolloutView) -> list[int]:
        trace = rollout.raw
        if trace.num_turns != 1:
            raise ValueError(
                f"opsd supports single-step trajectories only; "
                f"env '{rollout.env_name}' produced {trace.num_turns} model turn(s)."
            )
        demonstration = self._demonstration(rollout)
        template_fields = self._template_fields(rollout, demonstration)

        # The scoring prompt is the branch's leading non-sampled (input)
        # messages — the context the model conditioned on before responding.
        branch = trace.branches[0]
        messages = [node.message.model_dump(exclude_none=True) for node in branch.nodes if not node.sampled]
        return self._demo_conditioned_prefix_ids(
            messages, demonstration, rollout.env_name, template_fields=template_fields
        )

    def _validate_branch_sample_alignment(self, branch, sample, env_name: str) -> None:
        if list(sample.token_ids) != list(branch.token_ids):
            raise ValueError(
                f"opsd expected sample tokens to align with the trace branch "
                f"(env '{env_name}', sample={len(sample.token_ids)}, branch={len(branch.token_ids)})."
            )
        if len(sample.mask) != len(branch.sampled_mask):
            raise ValueError(
                f"opsd expected sample mask to align with the trace branch "
                f"(env '{env_name}', sample={len(sample.mask)}, branch={len(branch.sampled_mask)})."
            )

    async def score_batch(self, batch: list[RolloutView]) -> None:
        pool = self.teacher_pool
        assert pool is not None, "teacher pool not connected — Algorithm.setup() must run first"
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def score_completion(client, prefix_ids: list[int], completion_ids: list[int]) -> list[float]:
            async with semaphore:
                full_logprobs = await compute_prefill_logprobs(client, pool.model_name, prefix_ids + completion_ids)
            return full_logprobs[-len(completion_ids) :]

        async def score_single_turn(client, rollout: RolloutView) -> None:
            prefix_ids = self._ref_prefix_ids(rollout)
            assert len(rollout.samples) == 1  # single-step trajectory → one sample
            sample = rollout.samples[0]
            completion_ids = [t for t, trains in zip(sample.token_ids, sample.mask) if trains]
            completion_logprobs = await score_completion(client, prefix_ids, completion_ids)
            # Scatter the demo-conditioned completion logprobs back onto the
            # sample's trainable positions; full-length-N, 0.0 elsewhere.
            ref_logprobs = [0.0] * len(sample.token_ids)
            li = 0
            for i, trains in enumerate(sample.mask):
                if trains:
                    ref_logprobs[i] = completion_logprobs[li]
                    li += 1
            sample.ref_logprobs = ref_logprobs

        async def score_multi_turn(client, rollout: RolloutView) -> None:
            demonstration = self._demonstration(rollout)
            trainable_branches = [branch for branch in rollout.raw.branches if any(branch.sampled_mask)]
            if len(trainable_branches) != len(rollout.samples):
                raise ValueError(
                    f"opsd expected one sample per trainable branch; got {len(rollout.samples)} sample(s) "
                    f"for {len(trainable_branches)} branch(es) (env '{rollout.env_name}')."
                )
            for branch, sample in zip(trainable_branches, rollout.samples):
                self._validate_branch_sample_alignment(branch, sample, rollout.env_name)
                ref_logprobs = [0.0] * len(sample.token_ids)
                offset = 0
                for node_index, node in enumerate(branch.nodes):
                    span = range(offset, offset + len(node.token_ids))
                    positions = [i for i in span if sample.mask[i]]
                    if node.sampled and positions:
                        messages = [prior.message.model_dump(exclude_none=True) for prior in branch.nodes[:node_index]]
                        template_fields = self._template_fields(rollout, demonstration, branch, node_index)
                        prefix_ids = self._demo_conditioned_prefix_ids(
                            messages, demonstration, rollout.env_name, template_fields=template_fields
                        )
                        completion_ids = [sample.token_ids[i] for i in positions]
                        completion_logprobs = await score_completion(client, prefix_ids, completion_ids)
                        for i, logprob in zip(positions, completion_logprobs):
                            ref_logprobs[i] = logprob
                    offset += len(node.token_ids)
                sample.ref_logprobs = ref_logprobs

        async def score_one(client, rollout: RolloutView) -> None:
            if rollout.raw.num_turns == 1:
                await score_single_turn(client, rollout)
            elif self.multi_turn:
                await score_multi_turn(client, rollout)
            else:
                self._ref_prefix_ids(rollout)

        await asyncio.gather(*[score_one(client, rollout) for client, rollout in zip(cycle(pool.train_clients), batch)])
