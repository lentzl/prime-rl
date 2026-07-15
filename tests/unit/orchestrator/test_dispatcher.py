import asyncio
import uuid
from types import SimpleNamespace

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.types import GroupState


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
