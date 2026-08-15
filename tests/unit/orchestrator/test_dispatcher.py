import asyncio
import uuid

import pytest

from prime_rl.orchestrator.dispatcher import DispatcherMode, RolloutDispatcher
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import GroupState


def _dispatcher(*, scheduled: int, limit: int = 16) -> RolloutDispatcher:
    dispatcher = object.__new__(RolloutDispatcher)
    dispatcher.max_inflight = 8
    dispatcher.inflight_permits = 0
    dispatcher.train_rollouts_per_policy = limit
    dispatcher.train_rollouts_scheduled = scheduled
    dispatcher.groups = {}
    return dispatcher


def test_return_train_rollout_slots_reopens_synchronous_budget():
    dispatcher = _dispatcher(scheduled=16)

    dispatcher.return_train_rollout_slots(1)

    assert dispatcher.train_rollouts_scheduled == 15
    assert dispatcher.available_train_permits == 1


def test_return_train_rollout_slots_never_underflows():
    dispatcher = _dispatcher(scheduled=1)

    dispatcher.return_train_rollout_slots(4)

    assert dispatcher.train_rollouts_scheduled == 0


def test_return_train_rollout_slots_rejects_negative_count():
    dispatcher = _dispatcher(scheduled=16)

    with pytest.raises(ValueError, match="must be non-negative"):
        dispatcher.return_train_rollout_slots(-1)


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
    sink.train_rollout_replacements = 0

    sink.record_batch_admission(expected_rollouts=2, admitted_rollouts=1)
    sink.record_batch_admission(expected_rollouts=2, admitted_rollouts=3)

    assert sink.take_train_rollout_replacements() == 1
    assert sink.take_train_rollout_replacements() == 0
