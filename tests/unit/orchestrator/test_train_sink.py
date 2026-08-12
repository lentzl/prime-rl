import verifiers.v1 as vf

from prime_rl.orchestrator.metrics import TrainRollouts
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import Rollout
from prime_rl.transport import TrainingSample


def _rollout(weight: float) -> Rollout:
    sample = TrainingSample(
        token_ids=[1, 2],
        mask=[False, True],
        logprobs=[0.0, -0.1],
        temperatures=[1.0, 1.0],
        env_name="test-env",
        sdpo_weights=[0.0, weight],
    )
    rollout = Rollout(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[],
        rewards={},
        env_name="test-env",
    )
    rollout.samples = [sample]
    return rollout


def _sink(*rollouts: Rollout) -> TrainSink:
    sink = object.__new__(TrainSink)
    sink.batch_size = len(rollouts)
    sink.token_batch_size = None
    sink.pending_batch = list(rollouts)
    sink.pending_tokens = 0
    sink.pending_rollouts = TrainRollouts(list(rollouts))
    sink.post_filters = []
    return sink


def test_process_batch_excludes_rollouts_without_a_loss_signal() -> None:
    untrainable = _rollout(0.0)
    trainable = _rollout(1.0)

    batch = _sink(untrainable, trainable).process_batch()

    assert batch.samples == trainable.samples


def test_process_batch_does_not_ship_an_all_zero_update() -> None:
    untrainable = _rollout(0.0)
    sink = _sink(untrainable)

    batch = sink.process_batch()

    assert batch.samples == []
    assert list(batch.rollouts) == [untrainable]
    assert list(sink.pending_rollouts) == [untrainable]
