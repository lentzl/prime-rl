import asyncio
import uuid
from collections import deque
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.dispatcher import DispatcherMode, RolloutDispatcher
from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import GroupState


def _dispatcher(*, scheduled: int, limit: int = 16) -> RolloutDispatcher:
    dispatcher = object.__new__(RolloutDispatcher)
    dispatcher.max_inflight = 8
    dispatcher.inflight_permits = 0
    dispatcher.train_rollouts_per_policy = limit
    dispatcher.train_rollouts_scheduled = scheduled
    dispatcher.train_source_minimums = {}
    dispatcher.minimum_train_envs = deque()
    dispatcher.replacement_train_envs = deque()
    dispatcher.groups = {}
    return dispatcher


def test_return_train_rollout_slots_reopens_synchronous_budget():
    dispatcher = _dispatcher(scheduled=16)

    dispatcher.return_train_rollout_slots(1, ["parallel"])

    assert dispatcher.train_rollouts_scheduled == 15
    assert dispatcher.available_train_permits == 1
    assert list(dispatcher.replacement_train_envs) == ["parallel"]


def test_return_train_rollout_slots_never_underflows():
    dispatcher = _dispatcher(scheduled=1)

    dispatcher.return_train_rollout_slots(4)

    assert dispatcher.train_rollouts_scheduled == 0


def test_return_train_rollout_slots_rejects_negative_count():
    dispatcher = _dispatcher(scheduled=16)

    with pytest.raises(ValueError, match="must be non-negative"):
        dispatcher.return_train_rollout_slots(-1)


def test_satisfied_source_removes_queued_minimum_and_replacement_groups():
    dispatcher = _dispatcher(scheduled=12)
    dispatcher.minimum_train_envs.extend(["direct", "parallel", "direct"])
    dispatcher.replacement_train_envs.extend(["causal", "direct"])

    dispatcher.mark_train_sources_satisfied({"direct"})

    assert list(dispatcher.minimum_train_envs) == ["parallel"]
    assert list(dispatcher.replacement_train_envs) == ["causal"]


def test_open_train_group_can_finish_after_budget_is_exhausted():
    dispatcher = _dispatcher(scheduled=16)
    dispatcher.groups[uuid.uuid4()] = GroupState(
        kind="train",
        env_name="test",
        task=object(),
        rollouts_to_schedule=1,
        target_rollouts=2,
    )

    assert dispatcher.available_train_permits == 0
    assert dispatcher.has_open_train_group is True


def test_fill_inflight_finishes_open_group_after_budget_is_exhausted():
    dispatcher = _dispatcher(scheduled=16)
    group = GroupState(
        kind="train",
        env_name="test",
        task=object(),
        rollouts_to_schedule=1,
        target_rollouts=2,
    )
    dispatcher.groups[uuid.uuid4()] = group
    dispatcher.mode = DispatcherMode.PREFER_TRAIN
    dispatcher.dispatch_allowed = asyncio.Event()
    dispatcher.dispatch_allowed.set()
    calls = 0

    async def finish_group(kind):
        nonlocal calls
        assert kind == "train"
        calls += 1
        group.rollouts_to_schedule = 0
        return True

    dispatcher.try_schedule = finish_group

    asyncio.run(dispatcher.fill_inflight())

    assert calls == 1


def test_train_sink_records_only_batch_admission_deficit():
    sink = object.__new__(TrainSink)
    sink.batch_source_minimums = {}
    sink.train_rollout_replacements = 0
    sink.train_rollout_replacement_envs = []

    sink.record_batch_admission(env_name="parallel", expected_rollouts=2, admitted_rollouts=1)
    sink.record_batch_admission(env_name="direct", expected_rollouts=2, admitted_rollouts=3)

    assert sink.take_train_rollout_replacements() == (1, ["parallel"])
    assert sink.take_train_rollout_replacements() == (0, [])


def test_full_batch_reopens_only_missing_source_quota_once():
    sink = object.__new__(TrainSink)
    sink.batch_source_minimums = {"coordinator": 2, "causal": 4}
    sink.pending_batch = [SimpleNamespace(env_name="causal") for _ in range(16)]
    sink.train_rollout_replacements = 0
    sink.train_rollout_replacement_envs = []
    sink.source_replacements_outstanding = set()
    sink.train_envs = SimpleNamespace(
        get=lambda env_name: SimpleNamespace(config=SimpleNamespace(group_size=2))
    )

    sink.record_missing_source_minimum_replacements()
    sink.record_missing_source_minimum_replacements()

    assert sink.take_train_rollout_replacements() == (2, ["coordinator"])
    assert sink.source_replacements_outstanding == {"coordinator"}


def test_met_source_quota_suppresses_stale_group_replacement():
    sink = object.__new__(TrainSink)
    sink.batch_source_minimums = {"causal": 4}
    sink.pending_batch = [SimpleNamespace(env_name="causal") for _ in range(4)]
    sink.train_rollout_replacements = 0
    sink.train_rollout_replacement_envs = []
    sink.source_replacements_outstanding = {"causal"}
    sink.satisfied_train_sources = set()

    sink.record_batch_admission(env_name="causal", expected_rollouts=2, admitted_rollouts=1)

    assert sink.take_train_rollout_replacements() == (0, [])
    assert sink.take_satisfied_train_sources() == {"causal"}


def test_replacement_group_uses_the_source_that_lost_admission():
    dispatcher = _dispatcher(scheduled=15)
    dispatcher.policy = SimpleNamespace(version=0)
    requested = []

    class Source:
        def next_example(self, env_name=None):
            requested.append(env_name)
            return {"env_name": env_name, "task": object()}

    class Envs:
        def get(self, env_name):
            return SimpleNamespace(config=SimpleNamespace(group_size=2))

    dispatcher.train_source = Source()
    dispatcher.replacement_train_envs.append("parallel")

    group = dispatcher.next_fresh_group("train", Envs(), available_permits=1)

    assert group is not None and group.env_name == "parallel"
    assert requested == ["parallel"]
    assert not dispatcher.replacement_train_envs


def test_minimum_source_group_is_scheduled_before_weighted_sampling():
    dispatcher = _dispatcher(scheduled=0)
    dispatcher.policy = SimpleNamespace(version=0)
    requested = []

    class Source:
        def next_example(self, env_name=None):
            requested.append(env_name)
            return {"env_name": env_name or "weighted", "task": object()}

    class Envs:
        def get(self, env_name):
            return SimpleNamespace(config=SimpleNamespace(group_size=1))

    dispatcher.train_source = Source()
    dispatcher.minimum_train_envs.extend(["direct", "parallel"])

    first = dispatcher.next_fresh_group("train", Envs(), available_permits=8)
    second = dispatcher.next_fresh_group("train", Envs(), available_permits=8)
    weighted = dispatcher.next_fresh_group("train", Envs(), available_permits=8)

    assert [first.env_name, second.env_name, weighted.env_name] == ["direct", "parallel", "weighted"]
    assert requested == ["direct", "parallel", None]


def test_new_policy_reseeds_minimum_source_groups():
    dispatcher = _dispatcher(scheduled=16)
    dispatcher.train_source_minimums = {"direct": 1, "parallel": 3}
    dispatcher.train_envs = SimpleNamespace(
        get=lambda env_name: SimpleNamespace(config=SimpleNamespace(group_size=2))
    )
    dispatcher.minimum_train_envs.extend(["stale"])
    dispatcher.replacement_train_envs.extend(["stale"])

    asyncio.run(dispatcher.on_new_version(1))

    assert dispatcher.train_rollouts_scheduled == 0
    assert list(dispatcher.minimum_train_envs) == ["direct", "parallel", "parallel"]
    assert not dispatcher.replacement_train_envs


def test_train_sink_waits_for_source_minimum_and_selects_it_into_cohort():
    sink = object.__new__(TrainSink)
    sink.batch_size = 3
    sink.token_batch_size = None
    sink.batch_source_minimums = {"direct": 1}
    sink.post_filters = []

    def rollout(env_name):
        return SimpleNamespace(env_name=env_name, has_error=False, is_filtered=False, samples=[env_name])

    first = rollout("parallel")
    second = rollout("parallel")
    third = rollout("single")
    sink.pending_batch = [first, second, third]

    assert sink.source_minimums_met() is False

    direct = rollout("direct")
    sink.pending_batch.append(direct)
    sink.pending_rollouts = TrainRollouts(sink.pending_batch.copy())
    assert sink.source_minimums_met() is True

    batch = sink.process_batch()

    assert batch.samples == ["parallel", "parallel", "direct"]
    assert batch.rollouts.rollouts == [first, second, direct]
    assert sink.pending_batch == [third]
    assert sink.pending_rollouts.rollouts == [third]
