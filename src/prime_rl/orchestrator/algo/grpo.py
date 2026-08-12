from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty
        self.action_filter_fn = None
        if config.action_filter is not None:
            self.action_filter_fn = partial(
                import_object(config.action_filter.import_path),
                **config.action_filter.kwargs,
            )

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            advantages = rewards - rewards.mean()
        else:
            output = torch.tensor([rollout.num_output_tokens for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            self._assign_advantage(rollout, advantage)

    def _assign_advantage(self, rollout: Rollout, advantage: float) -> None:
        if self.action_filter_fn is None:
            rollout.assign_advantages(advantage)
            return

        trainable_branches = list(iter_trainable_branches(rollout))
        masks = self.action_filter_fn(rollout)
        if not isinstance(masks, list) or len(masks) != len(trainable_branches):
            got = len(masks) if isinstance(masks, list) else type(masks).__name__
            raise ValueError(
                "grpo action filter must return one keep-mask per trainable branch: "
                f"got {got}, expected {len(trainable_branches)}"
            )

        advantages: list[float] = []
        for branch_index, ((branch, trainable_mask), keep_mask) in enumerate(
            zip(trainable_branches, masks, strict=True)
        ):
            expected = len(branch.token_ids)
            if not isinstance(keep_mask, list) or len(keep_mask) != expected:
                got = len(keep_mask) if isinstance(keep_mask, list) else type(keep_mask).__name__
                raise ValueError(
                    f"grpo action filter mask for branch {branch_index} must span the branch's tokens: "
                    f"got {got}, expected {expected}"
                )
            advantages.extend(
                advantage if trainable and keep else 0.0
                for trainable, keep in zip(trainable_mask, keep_mask, strict=True)
            )
        rollout.assign_advantages(advantages)
