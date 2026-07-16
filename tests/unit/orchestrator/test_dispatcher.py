import asyncio
import uuid
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.types import GroupState


class FakeEnvs:
    def __init__(self, env):
        self.env = env

    def __iter__(self):
        return iter([self.env])

    def get(self, name):
        return self.env


def test_non_group_scoring_rollouts_can_fill_a_partial_group():
    async def run() -> None:
        blocked = asyncio.Event()

        async def run_rollout(**kwargs):
            await blocked.wait()
            return kwargs

        async def select_train_client(load):
            return object()

        pool = SimpleNamespace(select_train_client=select_train_client)
        env = SimpleNamespace(
            name="test-env",
            config=SimpleNamespace(group_size=8),
            requires_group_scoring=False,
            run_rollout=run_rollout,
            sampler=SimpleNamespace(samples_from_live_policy=True, pool=pool),
        )
        source = SimpleNamespace(next_example=lambda permits: {"env_name": "test-env", "task_idx": 0})
        dispatcher = RolloutDispatcher(
            train_envs=FakeEnvs(env),
            eval_envs=None,
            train_source=source,
            eval_source=None,
            policy_pool=pool,
            policy=SimpleNamespace(version=0, model_name="policy"),
            max_inflight_rollouts=4,
            train_rollouts_per_policy=None,
            tasks_per_minute=None,
            max_off_policy_steps=8,
        )

        await dispatcher.fill_inflight()

        assert dispatcher.inflight_permits == 4
        assert len(dispatcher.inflight) == 4
        group = next(iter(dispatcher.groups.values()))
        assert group.target_rollouts == 8
        assert group.rollouts_to_schedule == 4

        for task in dispatcher.inflight:
            task.cancel()
        await asyncio.gather(*dispatcher.inflight, return_exceptions=True)

    asyncio.run(run())


def test_group_scoring_requires_capacity_for_the_whole_group():
    env = SimpleNamespace(
        name="test-env",
        config=SimpleNamespace(group_size=8),
        requires_group_scoring=True,
    )

    with pytest.raises(ValueError, match="group-scoring env 'test-env'"):
        RolloutDispatcher(
            train_envs=FakeEnvs(env),
            eval_envs=None,
            train_source=object(),
            eval_source=None,
            policy_pool=object(),
            policy=SimpleNamespace(version=0),
            max_inflight_rollouts=4,
            train_rollouts_per_policy=None,
            tasks_per_minute=None,
            max_off_policy_steps=8,
        )


def test_synchronous_dispatch_budget_resets_only_for_new_policy():
    async def run() -> None:
        async def run_rollout(**kwargs):
            return kwargs

        env = SimpleNamespace(requires_group_scoring=False, run_rollout=run_rollout)
        dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
        dispatcher.train_envs = SimpleNamespace(get=lambda name: env)
        dispatcher.eval_envs = None
        dispatcher.policy = SimpleNamespace(version=0)
        dispatcher.max_inflight = 16
        dispatcher.inflight_permits = 0
        dispatcher.inflight = {}
        dispatcher.groups = {}
        dispatcher.rate_limiter = None
        dispatcher.train_rollouts_per_policy = 16
        dispatcher.train_rollouts_scheduled = 15
        dispatcher._train_pool_for = lambda name: (None, "policy", True)

        group_id = uuid.uuid4()
        group = GroupState(
            kind="train",
            env_name="test-env",
            task_idx=0,
            rollouts_to_schedule=1,
            target_rollouts=16,
            pinned_client=object(),
            policy_version_at_start=0,
        )
        dispatcher.groups[group_id] = group

        assert await dispatcher.schedule_group_rollout(group_id, group)
        assert dispatcher.train_rollouts_scheduled == 16
        assert dispatcher.available_train_permits == 0

        task = next(iter(dispatcher.inflight))
        await task
        await dispatcher.on_new_version(1)

        assert dispatcher.train_rollouts_scheduled == 0
        assert dispatcher.available_train_permits == 15

    asyncio.run(run())
